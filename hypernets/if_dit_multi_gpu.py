# The purpose of this script is to train the image field diffusion transformer
# on multiple GPUs, with data sharding across the batch dimension.
import jax
from jax import lax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali import types as dali_types

from functools import partial
from typing import Optional, Callable, Any
import json
import os

from hypernets.common.config_utils import load_activation_fn, load_dtype
from hypernets.packing.alt_ngp import unflatten_params
from fields import ngp_image
from matplotlib import pyplot as plt
import wandb

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float = 1000.0
    embedding_min_frequency:float = 1.0
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(self.embedding_min_frequency),
                jnp.log(self.embedding_max_frequency),
                self.embedding_dim // 2,
                dtype=self.dtype
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        embeddings = jnp.concatenate(
            [jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)],
            axis = -1,
            dtype=self.dtype
        )
        return embeddings

class FeedForward(nn.Module):
    hidden_dim:int
    output_dim:int
    activation_fn:Callable = nn.gelu
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype)(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        x = self.activation_fn(x)
        return x

class VanillaTransformer(nn.Module):
    num_heads:int
    num_blocks:int
    attention_dim:int
    residual_dim:int
    feed_forward_dim:int
    activation_fn:Callable
    remat:bool = False
    dtype:Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        if self.remat:
            CustomAttention = nn.remat(nn.SelfAttention)
            CustomFeedForward = nn.remat(FeedForward)
        else:
            CustomAttention = nn.SelfAttention
            CustomFeedForward = FeedForward
        
        for _ in range(self.num_blocks):
            residual = x
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x = CustomAttention(
                num_heads=self.num_heads,
                qkv_features=self.attention_dim, 
                out_features=self.residual_dim,
                dtype=self.dtype
            )(x)
            x = x + residual
            residual = x
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x = CustomFeedForward(
                self.feed_forward_dim, 
                self.residual_dim, 
                activation_fn=self.activation_fn,
                dtype=self.dtype
            )(x)
            x = x + residual
        return x

class DiffusionTransformer(nn.Module):
    attention_dim:int
    num_attention_heads:int
    embedding_dim:int
    num_blocks:int
    feed_forward_dim:int
    token_dim:int
    context_length:int
    activation_fn:Callable
    dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x, t):
        positions = jnp.arange(self.context_length+1)
        position_embedding = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.dtype
        )(positions)
        position_embedding = jnp.expand_dims(position_embedding, axis=0)
        
        time_embedding = SinusoidalEmbedding(
            self.embedding_dim,
            dtype=self.dtype
        )(t)
        time_embedding = nn.Dense(
            self.embedding_dim, 
            dtype=self.dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.Dense(
            self.embedding_dim,
            dtype=self.dtype
        )(time_embedding)
        time_embedding = self.activation_fn(time_embedding)
        time_embedding = nn.LayerNorm()(time_embedding)
        
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.Dense(
            self.embedding_dim, 
            dtype=self.dtype
        )(x)
        x = self.activation_fn(x)
        x = nn.LayerNorm()(x)

        # Add the diffusion time token to the end of the sequence.
        time_embedding = jnp.expand_dims(time_embedding, axis=1)
        x = jnp.concatenate([x, time_embedding], axis=-2)
        x = x + position_embedding

        x = VanillaTransformer(
            num_blocks=self.num_blocks,
            attention_dim=self.attention_dim,
            num_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim,
            feed_forward_dim=self.feed_forward_dim,
            activation_fn=self.activation_fn,
            dtype=self.dtype,
            remat=self.remat
        )(x)

        # Remove the diffusion time token from the end of the sequence.
        x = x[:, :-1, :]
        x = nn.Dense(
            features=self.token_dim, dtype=self.dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(x)
        return x

def diffusion_schedule(t):
    start_angle = jnp.arccos(0.999)
    end_angle = jnp.arccos(0.001)

    diffusion_angles = start_angle + t * (end_angle - start_angle)

    signal_rates = jnp.cos(diffusion_angles)
    noise_rates = jnp.sin(diffusion_angles)
    return noise_rates, signal_rates

def ddim_sample(
    state, 
    num_samples:int, 
    diffusion_steps:int, 
    token_dim:int, 
    context_length:int,
    seed:int 
):
    @jax.jit
    def inference_fn(state, noisy_batch, diffusion_times):
        return jax.lax.stop_gradient(
            state.apply_fn({'params': state.params}, noisy_batch, diffusion_times)
        )
    
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(seed), 
        shape=(num_samples, context_length, token_dim)
    )
    step_size = 1.0 / diffusion_steps
    
    next_noisy_batch = initial_noise
    for step in range(diffusion_steps):
        noisy_batch = next_noisy_batch
        
        diffusion_times = jnp.ones((num_samples, 1)) - step * step_size
        noise_rates, signal_rates = diffusion_schedule(diffusion_times)
        pred_noises = inference_fn(state, noisy_batch, noise_rates**2)
        pred_batch = (
            (noisy_batch - jnp.expand_dims(noise_rates, axis=1) * pred_noises) 
            / jnp.expand_dims(signal_rates, axis=1)
        )
        
        next_diffusion_times = diffusion_times - step_size
        next_noise_rates, next_signal_rates = diffusion_schedule(next_diffusion_times)
        next_noisy_batch = (
            jnp.expand_dims(next_signal_rates, axis=1) * pred_batch 
            + jnp.expand_dims(next_noise_rates, axis=1) * pred_noises
        )
    return pred_batch

def train_step(state, batch, batch_size):
    noise_key, diffusion_time_key = jax.random.split(jax.random.PRNGKey(state.step), 2)
    noises = jax.random.normal(noise_key, batch.shape, dtype=jnp.float32)
    diffusion_times = jax.random.uniform(diffusion_time_key, (batch_size, 1))
    noise_rates, signal_rates = diffusion_schedule(diffusion_times)
    noisy_batch = (
        jnp.expand_dims(signal_rates, axis=1) * batch 
        + jnp.expand_dims(noise_rates, axis=1) * noises
    )

    def loss_fn(params):
        pred_noises = state.apply_fn({'params': params}, noisy_batch, noise_rates**2)
        return jnp.mean((pred_noises - noises)**2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def train_loop(
    state, train_step_fn, num_epochs, steps_per_epoch, data_iterator, 
    batch_size, context_length, token_dim, output_dir,
    field_state, image_width, image_height, field_param_map
):
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for step in range(steps_per_epoch):
            batch = next(data_iterator)['x']
            loss, state = train_step_fn(state, batch)
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        #print(average_loss)
        wandb.log({'loss': average_loss}, state.step)
        samples = ddim_sample(
            state=state,
            num_samples=10,
            diffusion_steps=20,
            token_dim=token_dim,
            context_length=context_length,
            seed=0
        )
        for i, sample in enumerate(samples):
            field_params = unflatten_params(flat_params=sample, param_map=field_param_map)
            field_state = field_state.replace(params=field_params)
            field_render = ngp_image.render_image(field_state, image_height, image_width)
            field_render = jax.device_put(field_render, jax.devices('cpu')[0])
            plt.imsave(os.path.join(output_dir, f'epoch{epoch}_image{i}.png'), field_render)

def init_state(key, x_shape, t_shape, model, optimizer, input_sharding, mesh):
    def init_fn(key, x, t, model, optimizer):
        params = model.init(key, x, t)['params']
        state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        return state
    
    x = jax.device_put(jnp.ones(x_shape), input_sharding)
    t = jax.device_put(jnp.ones(t_shape), input_sharding)

    state_sharding = nn.get_sharding(
        jax.eval_shape(partial(init_fn, model=model, optimizer=optimizer), key, x, t), 
        mesh
    )
    state = jax.jit(
        init_fn, static_argnames=('model', 'optimizer'),
        in_shardings=(NamedSharding(mesh, PartitionSpec(None)), input_sharding, input_sharding),
        out_shardings=state_sharding
    )(key, x, t, model, optimizer)
    return state, state_sharding

def get_data_iterator(dataset_path, token_dim, batch_size, input_sharding, num_threads=4):
    dataset_list = os.listdir(dataset_path)
    valid_dataset_list = [path for path in dataset_list if path.endswith('.npy')]
    assert len(valid_dataset_list) > 0, f'Could not find any .npy files in {dataset_path}'
    
    dummy_sample = jnp.load(os.path.join(dataset_path, valid_dataset_list[0]))
    print('Sample shape:', dummy_sample.shape)
    context_length = int(dummy_sample.shape[0] / token_dim)

    @pipeline_def
    def my_pipeline_def(file_root, context_length, token_dim):
        numpy_data = fn.readers.numpy(
            device='cpu', 
            file_root=file_root, 
            file_filter='*.npy', 
            shuffle_after_epoch=True,
            name='r'
        )
        numpy_data = numpy_data.gpu()
        tokens = fn.reshape(
            numpy_data, 
            shape=[context_length, token_dim], 
            device='gpu'
        )
        return tokens
    
    data_pipeline = my_pipeline_def(
        file_root=dataset_path,
        batch_size=batch_size,
        context_length=context_length,
        token_dim=token_dim,
        num_threads=num_threads,
        device_id=0
    )
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], 
        output_map=['x'], 
        last_batch_policy=LastBatchPolicy.DROP,
        sharding=input_sharding
    )
    num_batches = len(valid_dataset_list) // batch_size
    return data_iterator, num_batches, context_length

def main():
    devices = jax.devices('gpu')
    num_devices = len(devices)
    print(f'Found {num_devices} GPU(s)')
    print(devices)

    output_dir = 'data/if_dit_runs/3'
    dataset_path = 'data/mnist_ingp_flat'
    config_path = 'configs/if_dit_multi_gpu.json'
    field_config_path = 'configs/ngp_image.json'
    num_epochs = 1000
    # These are the dimensions of the images encoded by the neural fields.
    # The neural field dataset is trained on MNIST so the dimensions are 28x28.
    image_width = 28
    image_height = 28
   
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(field_config_path, 'r') as f:
        field_config = json.load(f)
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        field_param_map = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)
    config['num_devices'] = num_devices
    token_dim = config['token_dim']
    batch_size = config['batch_size']

    device_mesh = mesh_utils.create_device_mesh((num_devices,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    input_sharding = NamedSharding(mesh, PartitionSpec('data'))

    data_iterator, steps_per_epoch, context_length = \
        get_data_iterator(dataset_path, token_dim, batch_size, input_sharding)
    print('Token dim:', token_dim)
    print('Context length:', context_length)
    print('Steps per epoch:', steps_per_epoch)
    
    model = DiffusionTransformer(
        attention_dim=config['attention_dim'],
        num_attention_heads=config['num_attention_heads'],
        embedding_dim=config['embedding_dim'],
        feed_forward_dim=config['feed_forward_dim'],
        num_blocks=config['num_blocks'],
        token_dim=token_dim,
        context_length=context_length,
        activation_fn=load_activation_fn(config['activation_fn']),
        dtype=load_dtype(config['dtype']),
        remat=config['remat']
    )
    
    opt = optax.chain(
        optax.zero_nans(),
        optax.adaptive_grad_clip(clipping=config['grad_clip']),
        optax.adamw(
            learning_rate=config['learning_rate'], 
            weight_decay=config['weight_decay']
        )
    )
    state, state_sharding = init_state(
        key=jax.random.PRNGKey(11), 
        x_shape=(batch_size, context_length, token_dim),
        t_shape=(batch_size, 1),
        model=model,
        optimizer=opt,
        input_sharding=input_sharding,
        mesh=mesh,
    )

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    config['param_count'] = param_count
    print('Param count:', param_count)
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(
        model=field_model, learning_rate=1e-3, KEY=jax.random.PRNGKey(2)
    )

    jit_train_step = jax.jit(
        partial(train_step, batch_size=batch_size), 
        in_shardings=(state_sharding, input_sharding),
        out_shardings=(None, state_sharding)
    )

    wandb.init(project='if-dit-r', config=config)
    train_loop(
        state=state, train_step_fn=jit_train_step, num_epochs=num_epochs, 
        steps_per_epoch=steps_per_epoch, data_iterator=data_iterator, 
        batch_size=batch_size, context_length=context_length, 
        token_dim=token_dim, output_dir=output_dir, field_state=field_state,
        image_width=image_width, image_height=image_height, field_param_map=field_param_map
    )

if __name__ == '__main__':
    main()
