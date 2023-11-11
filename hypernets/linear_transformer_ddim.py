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

class LinearTransformerDDIM(nn.Module):
    attention_dim:int
    num_attention_heads:int
    token_dim:int
    embedding_dim:int
    num_bocks:int
    feed_forward_dim:int
    embedding_max_frequency:float
    context_length:int
    normal_dtype:Any
    quantized_dtype:Any

    @nn.compact
    def __call__(self, x):
        x, noise_variances = x
        e = SinusoidalEmbedding(
            self.token_dim, 
            self.embedding_max_frequency,
            dtype=self.quantized_dtype
        )(noise_variances)
        x = jnp.concatenate([x, e], axis=-2)
        x = nn.remat(nn.Dense)(features=self.embedding_dim, dtype=self.quantized_dtype)(x)
        positions = jnp.arange(self.context_length+1)
        e = nn.Embed(
            num_embeddings=self.context_length+1, 
            features=self.embedding_dim,
            dtype=self.quantized_dtype
        )(positions)
        x = x + e

        x = LinearTransformer(
            num_blocks=self.num_bocks, 
            attention_dim=self.attention_dim, 
            num_attention_heads=self.num_attention_heads,
            residual_dim=self.embedding_dim, 
            feed_forward_dim=self.feed_forward_dim,
            quantized_dtype=self.quantized_dtype,
            normal_dtype=self.normal_dtype
        )(x)

        x = nn.remat(nn.Dense)(features=self.token_dim, dtype=self.quantized_dtype)(x)
        x = x[:, :-1, :] # Remove embedded noise variances token.
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

def tabulate(model, context_length, token_dim, dtype):
    tabulation = model.tabulate(
        jax.random.PRNGKey(0), 
        (
            jnp.ones((1, context_length, token_dim), dtype=dtype), 
            jnp.ones((1, 1, 1), dtype=dtype)
        )
    )
    print(tabulation)

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
    parser.add_argument('--checkpoint_step', type=int, default=0)
    parser.add_argument('--render_only', type=bool, default=False)
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    args = parser.parse_args()

    normal_dtype = jnp.float32
    quantized_dtype = jnp.float32
    batch_size = 64
    learning_rate = 5e-5
    datast_path = 'data/ngp_images/packed_ngp_anime_faces'
    all_sample_paths = os.listdir(datast_path)
    valid_sample_paths = []
    for path in all_sample_paths:
        if path.endswith('.npy'):
            full_path = os.path.join(datast_path, path)
            valid_sample_paths.append(full_path)
    del all_sample_paths
    load_batch_fn = partial(
        load_batch, 
        sample_paths=valid_sample_paths, 
        batch_size=batch_size,
        dtype=quantized_dtype
    )

    dummy_batch = load_batch_fn()
    print('batch shape', dummy_batch.shape)
    token_dim = dummy_batch.shape[-1]
    context_length = dummy_batch.shape[-2]
    print('context', context_length)

    model = LinearTransformerDDIM(
        attention_dim=512,
        num_attention_heads=8,
        token_dim=token_dim,
        embedding_dim=128,
        num_bocks=4,
        feed_forward_dim=128,
        embedding_max_frequency=1000.0,
        context_length=context_length,
        normal_dtype=normal_dtype,
        quantized_dtype=quantized_dtype
    )
    #tabulate(model, context_length, token_dim, quantized_dtype)

    tx = optax.adam(learning_rate)
    rng = jax.random.PRNGKey(0)
    x = (jnp.ones((1, context_length, token_dim)), jnp.ones((1, 1, 1)))
    params = model.init(rng, x)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if args.checkpoint_step > 0:
        checkpoint_name = f'data/generations/checkpoint_step_{args.checkpoint_step}'
        state = checkpointer.restore(checkpoint_name, item=state)
        print('Loaded checkpoint')
    if args.render_only:
        print('Render only mode')
        generated_weights = reverse_diffusion(
            state.apply_fn, 
            state.params, 
            num_images=1, 
            diffusion_steps=20, 
            context_length=context_length,
            token_dim=token_dim,
            diffusion_schedule_fn=diffusion_schedule,
            min_signal_rate=0.02,
            max_signal_rate=0.95,
            seed=2
        )
        print('Generated weights max:', jnp.max(generated_weights))
        print('Generated weights min:', jnp.min(generated_weights))
        rendered_image = unpack_and_render_ngp_image(
            config_path='configs/ngp_image.json',
            weight_map_path='data/ngp_images/packed_ngp_anime_faces/weight_map.json',
            packed_weights=generated_weights[0],
            image_width=64,
            image_height=64
        )
        plt.imsave(f'data/generations/render_only.png', rendered_image)
        exit(0)

    for step in range(args.checkpoint_step, args.train_steps+args.checkpoint_step):
        step_key = jax.random.PRNGKey(step)
        batch = load_batch_fn()
        loss, state = train_step(state, step_key, batch)
        if step % 1000 == 0 and step > args.checkpoint_step:
            print('Step:', step, 'Loss:', loss)
            generated_weights = reverse_diffusion(
                state.apply_fn, 
                state.params, 
                num_images=1, 
                diffusion_steps=20, 
                context_length=context_length,
                token_dim=token_dim,
                diffusion_schedule_fn=diffusion_schedule,
                min_signal_rate=0.02,
                max_signal_rate=0.95,
                seed=step
            )
            checkpointer.save(f'data/generations/checkpoint_step_{step}', state)
            print('Generated weights max:', jnp.max(generated_weights))
            print('Generated weights min:', jnp.min(generated_weights))
            rendered_image = unpack_and_render_ngp_image(
                config_path='configs/ngp_image.json',
                weight_map_path='data/ngp_images/packed_ngp_anime_faces/weight_map.json',
                packed_weights=generated_weights[0],
                image_width=64,
                image_height=64
            )
            plt.imsave(f'data/generations/ngp_image_step_{step}.png', rendered_image)
    print('Finished training')

if __name__ == '__main__':
    main()