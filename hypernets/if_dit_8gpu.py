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

class SinusoidalEmbedding(nn.Module):
    embedding_dim:int
    embedding_max_frequency:float
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
    )
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], 
        output_map=['x'], 
        last_batch_policy=LastBatchPolicy.DROP,
        sharding=input_sharidng
    )
    num_batches = len(valid_dataset_list) // batch_size
    return data_iterator, num_batches, context_length

def main():
    dataset_path = 'data/mnist_ingp_flat'
    # These are the dimensions of the images encoded by the neural fields.
    # The neural field dataset is trained on MNIST so the dimensions are 28x28.
    image_width = 28
    image_height = 28
   
    context_length = 128
    token_dim = 1
    batch_size = 32
    num_batches = 40
    x_train = jax.random.normal(
        jax.random.PRNGKey(1212), 
        (num_batches, batch_size, context_length, token_dim)
    )

    device_mesh = mesh_utils.create_device_mesh((8,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    input_sharding = NamedSharding(mesh, PartitionSpec('data'))

    data_iterator, steps_per_epoch, context_length = \
        get_data_iterator(dataset_path, token_dim, batch_size, input_sharding)

    model = DiffusionTransformer(
        attention_dim=128,
        num_attention_heads=4,
        embedding_dim=16,
        num_blocks=2,
        feed_forward_dim=32,
        context_length=context_length,
        token_dim=token_dim,
        activation_fn=nn.relu,
        dtype=jnp.float32,
        remat=False
    )
    opt = optax.adam(learning_rate=3e-4)
    state, state_sharding = init_state(
        key=jax.random.PRNGKey(11), 
        x_shape=(batch_size, context_length, token_dim),
        t_shape=(batch_size, 1),
        model=model,
        optimizer=opt,
        input_sharding=input_sharding,
        mesh=mesh,
    )

    batch = x_train[0]
    batch = jax.device_put(batch, input_sharding)
    jit_train_step = jax.jit(
        partial(train_step, batch_size=batch_size), 
        in_shardings=(state_sharding, input_sharding),
        out_shardings=(None, state_sharding)
    )
    loss, state = jit_train_step(state, batch)
    print('loss:', loss)

if __name__ == '__main__':
    main()
