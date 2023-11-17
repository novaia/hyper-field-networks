from typing import Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt
import os
import random
from functools import partial
from hypernets.common.nn import SinusoidalEmbedding, LinearTransformer
from hypernets.common.diffusion import diffusion_schedule, reverse_diffusion
from hypernets.common.rendering import unpack_and_render_ngp_image
import argparse
import orbax.checkpoint as ocp
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator

class LinearTransformerDDIM(nn.Module):
    attention_dim:int
    num_attention_heads:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int
    hash_table_height:int
    normal_dtype:Any
    quantized_dtype:Any
    remat:bool

    @nn.compact
    def __call__(self, x):
        CustomDense = nn.remat(nn.Dense) if self.remat else nn.Dense
        x, noise_variances = x
        e = SinusoidalEmbedding(
            self.token_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(noise_variances)
        x = jnp.concatenate([x, e], axis=-2)
        x = CustomDense(features=self.embedding_dim, dtype=self.quantized_dtype)(x)
        positions = jnp.arange(self.context_length+1)
        e = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        x = x + e

        x = LinearTransformer(
            num_blocks=self.num_bocks//2, 
            attention_dim=self.attention_dim, 
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype,
            remat=self.remat
        )(x)
        noise_variance_token = x[:, -1:, :]
        hash_table = x[:, :self.hash_table_height, :]
        hash_table = jnp.concatenate([hash_table, noise_variance_token], axis=-2)
        hash_table = LinearTransformer(
            num_blocks=self.num_bocks//2, 
            attention_dim=self.attention_dim, 
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype,
            remat=self.remat
        )(hash_table)
        hash_table = hash_table[:, :-1, :]
        hash_table = CustomDense(
            features=self.token_dim, dtype=self.quantized_dtype,
            kernel_init=nn.initializers.zeros_init()
        )(hash_table)

        network = x[:, self.hash_table_height:-1, :]
        network = jnp.concatenate([network, noise_variance_token], axis=-2) 
        network = LinearTransformer(
            num_blocks=self.num_bocks//2, 
            attention_dim=self.attention_dim, 
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype,
            remat=self.remat
        )(network)
        network = network[:, :-1, :]
        network = CustomDense(
            features=self.token_dim, dtype=self.quantized_dtype, 
            kernel_init=nn.initializers.zeros_init()
        )(network)

        x = jnp.concatenate([hash_table, network], axis=-2)
        return x

@jax.jit
def train_step(state:TrainState, key:int, batch:jax.Array):
    noise_key, diffusion_time_key = jax.random.split(key)

    def loss_fn(params):
        diffusion_times = jax.random.uniform(diffusion_time_key, (batch.shape[0], 1, 1))
        noise_rates, signal_rates = diffusion_schedule(diffusion_times, 0.02, 0.95)
        noise = jax.random.normal(noise_key, batch.shape)
        noisy_batch = batch * signal_rates + noise * noise_rates
        x = (noisy_batch, noise_rates**2)
        pred_noise = state.apply_fn({'params': params}, x)
        return jnp.mean((pred_noise - noise)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    grad = jax.tree_util.tree_map(jnp.nan_to_num, grad)
    state = state.apply_gradients(grads=grad)
    return loss, state

def load_batch(sample_paths, batch_size, dtype):
    random.shuffle(sample_paths)
    batch_paths = sample_paths[:batch_size]
    batch = []
    for path in batch_paths:
        batch.append(jnp.load(path))
    batch = jnp.array(batch, dtype=dtype)
    return batch

def get_data_iterator(dataset_path, batch_size, num_threads=3):
    @pipeline_def
    def my_pipeline_def():
        data = fn.readers.numpy(
            device='cpu', 
            file_root=dataset_path, 
            file_filter='*.npy', 
            random_shuffle=True,
            name='weight_reader'
        )
        return data

    my_pipeline = my_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    iterator = DALIGenericIterator(
        pipelines=[my_pipeline], output_map=['x'], reader_name='weight_reader'
    )
    return iterator

def inspect_intermediates(state, context_length, token_dim, dtype):
    x = (
        jnp.ones((1, context_length, token_dim), dtype=dtype), 
        jnp.ones((1, 1, 1), dtype=dtype)
    )
    _, intermediates = state.apply_fn({'params': state.params}, x, capture_intermediates=True)
    intermediates = intermediates['intermediates']
    print(intermediates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_epoch', type=int, default=-1)
    parser.add_argument('--render_only', type=bool, default=False)
    parser.add_argument('--train_epochs', type=int, default=1000)
    args = parser.parse_args()

    normal_dtype = jnp.float32
    quantized_dtype = jnp.float32
    batch_size = 16
    learning_rate = 5e-5
    diffusion_steps = 20
    image_width = 32
    image_height = 32
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    attention_dim = 512
    num_attention_heads = 8
    embedding_dim = 128
    num_blocks = 4
    feed_forward_dim = 128
    embedding_max_frequency = 1000.0
    hash_table_height = 48
    remat = False
    num_render_only_images = 5
    num_train_preview_images = 5

    checkpoint_path = 'data/cifar10_training4'
    dataset_path = 'data/ngp_images/packed_ngp_cifar10'
    config_path = 'configs/ngp_image.json'
    weight_map_path = os.path.join(dataset_path, 'weight_map.json')

    data_iterator = get_data_iterator(dataset_path, batch_size)
    dummy_batch = data_iterator.next()['x']
    print('Batch shape', dummy_batch.shape)
    token_dim = dummy_batch.shape[-1]
    context_length = dummy_batch.shape[-2]

    model = LinearTransformerDDIM(
        attention_dim=attention_dim,
        num_attention_heads=num_attention_heads,
        token_dim=token_dim,
        embedding_dim=embedding_dim,
        num_bocks=num_blocks,
        feed_forward_dim=feed_forward_dim,
        embedding_max_frequency=embedding_max_frequency,
        context_length=context_length,
        hash_table_height=hash_table_height,
        normal_dtype=normal_dtype,
        quantized_dtype=quantized_dtype,
        remat=remat
    )

    tx = optax.adam(learning_rate)
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones((1, context_length, token_dim)), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.checkpoint_epoch > -1:
        checkpoint_name = f'checkpoint_epoch_{args.checkpoint_epoch}'
        state = checkpointer.restore(os.path.join(checkpoint_path, checkpoint_name), item=state)
        print(f'Loaded checkpoint {checkpoint_name}')
    if args.render_only:
        print('Render only mode')
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_render_only_images, 
            diffusion_steps=diffusion_steps, 
            context_length=context_length,
            token_dim=token_dim,
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            seed=2
        )
        print('Generated weights max:', jnp.max(generated_weights))
        print('Generated weights min:', jnp.min(generated_weights))
        for i in range(num_render_only_images):
            rendered_image = unpack_and_render_ngp_image(
                config_path=config_path,
                weight_map_path=weight_map_path,
                packed_weights=generated_weights[i],
                image_width=image_width,
                image_height=image_height
            )
            plt.imsave(os.path.join(checkpoint_path, f'render_only{i}.png'), rendered_image)
        exit(0)

    gpu = jax.devices('gpu')[0]
    for epoch in range(args.checkpoint_epoch+1, args.checkpoint_epoch+args.train_epochs+1):
        losses_this_epoch = []
        for step, batch in enumerate(data_iterator):
            step_key = jax.random.PRNGKey(state.step)
            batch = jax.device_put(batch['x'], gpu)
            loss, state = train_step(state, step_key, batch)
            losses_this_epoch.append(loss)

        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print('Epoch:', epoch, 'Loss:', average_loss)
        checkpointer.save(os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}'), state)
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=num_train_preview_images, 
            diffusion_steps=diffusion_steps, 
            context_length=context_length,
            token_dim=token_dim,
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=min_signal_rate,
            max_signal_rate=max_signal_rate,
            seed=step
        )
        print('Generated weights max:', jnp.max(generated_weights))
        print('Generated weights min:', jnp.min(generated_weights))
        
        for i in range(num_train_preview_images):
            rendered_image = unpack_and_render_ngp_image(
                config_path=config_path,
                weight_map_path=weight_map_path,
                packed_weights=generated_weights[0],
                image_width=image_width,
                image_height=image_height
            )
            plt.imsave(
                os.path.join(checkpoint_path, f'image_{i}_epoch_{epoch}.png'), 
                rendered_image
            )
    print('Finished training')

if __name__ == '__main__':
    main()