import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from hypernets.common.nn import LinearAttention
import jax
import optax
import wandb
import os
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List

def get_data_iterator(dataset_path, batch_size, num_threads=3):
    @pipeline_def
    def my_pipeline_def():
        data = fn.readers.numpy(
            device='cpu', 
            file_root=dataset_path, 
            file_filter='*.npy', 
            shuffle_after_epoch=True,
            name='r'
        )
        return data
    my_pipeline = my_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    iterator = DALIGenericIterator(pipelines=[my_pipeline], output_map=['x'], reader_name='r')
    return iterator

def tokenize_batch(token_dim, batch):
    context_length = int(jnp.ceil(batch.shape[1] / token_dim))
    batch = jnp.resize(batch, (batch.shape[0], context_length, token_dim))
    return batch

def detokenize_batch(original_length, batch):
    batch = jnp.resize(batch, (batch.shape[0], original_length))
    return batch

def sample_normals(means, standard_deviations, key):
    def sample(mean, std, key):
        return std * jax.random.normal(key) + mean
    split_keys = jax.random.split(key, means.shape[-1])
    return jax.vmap(sample, in_axes=0)(means, standard_deviations, split_keys)

class Encoder(nn.Module):
    attention_dim: int
    context_length: int
    attention_out_dim: int
    num_attention_heads: int
    feed_forward_dim: int
    final_dim: int

    @nn.compact
    def __call__(self, x):
        positions = jnp.arange(self.context_length)
        e = nn.Embed(
            num_embeddings=self.context_length,
            features=x.shape[-1]
        )(positions)
        x = x + e
        x = LinearAttention(
            attention_dim=self.attention_dim, 
            output_dim=self.attention_out_dim,
            num_heads=self.num_attention_heads
        )(x)
        x = nn.Dense(features=self.feed_forward_dim)(x)
        x = nn.gelu(x)
        means = nn.Dense(features=self.final_dim)(x)
        deviations = nn.Dense(features=self.final_dim)(x)
        return means, deviations
    
class Decoder(nn.Module):
    attention_dim: int
    context_length: int
    attention_out_dim: int
    num_attention_heads: int
    feed_forward_dim: int
    final_dim: int

    @nn.compact
    def __call__(self, x):
        positions = jnp.arange(self.context_length)
        e = nn.Embed(
            num_embeddings=self.context_length,
            features=x.shape[-1]
        )(positions)
        x = x + e
        x = LinearAttention(
            attention_dim=self.attention_dim, 
            output_dim=self.attention_out_dim,
            num_heads=self.num_attention_heads
        )(x)
        x = nn.Dense(features=self.feed_forward_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.final_dim)(x)
        return x
    
class Vae(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x):
        x, key = x
        means, deviations = self.encoder(x)
        x = jax.vmap(
            jax.vmap(sample_normals, in_axes=(0, 0, None)), in_axes=(0, 0, None)
        )(means, deviations, key)
        x = self.decoder(x)
        x = nn.sigmoid(x)
        return x, means, deviations
    
class BaselineVaeEncoder(nn.Module):
    widths: List[int]
    output_width: int

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.Dense(width)(x)
            x = nn.gelu(x)
        means = nn.Dense(self.output_width)(x)
        deviations = nn.Dense(self.output_width)(x)
        deviations = jnp.exp(deviations)
        return means, deviations

class BaselineVaeDecoder(nn.Module):
    widths: List[int]
    output_width: int

    @nn.compact
    def __call__(self, x):
        for width in self.widths:
            x = nn.Dense(width)(x)
            x = nn.gelu(x)
        x = nn.Dense(self.output_width)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_width)(x)
        return x

class BaselineVae(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x):
        x, key = x
        means, deviations = self.encoder(x)
        x = means + deviations * jax.random.normal(key, means.shape)
        x = self.decoder(x)
        return x, means, deviations

@jax.jit
def train_step(state, batch, kl_weight):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits, means, deviations = state.apply_fn({'params': params}, [batch, key])
        bce_loss = jnp.sum(optax.sigmoid_binary_cross_entropy(logits, batch))
        kld_loss = jnp.sum(deviations**2 + means**2 - jnp.log(deviations) - 0.5)
        loss = bce_loss * (1 - kl_weight) + kld_loss * (kl_weight)
        return loss, (bce_loss, kld_loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (bce_loss, kld_loss)), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return (loss, bce_loss, kld_loss), state

def train_baseline():
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 64
    num_generative_samples = 3
    kl_weight = 0.5
    input_dim = 784
    image_width = 28
    image_height = 28
    latent_dim = 2
    dataset_path = 'data/easy-mnist/mnist_numpy_flat/data'
    encoder_widths = [500, 500]
    # It is necessary to cast to a list here or else Flax will not retain the reversed view.
    decoder_widths = list(reversed(encoder_widths))
    encoder = BaselineVaeEncoder(encoder_widths, latent_dim)
    decoder = BaselineVaeDecoder(decoder_widths, input_dim)
    model = BaselineVae(encoder, decoder)

    key = jax.random.PRNGKey(0)
    x = [jnp.ones((batch_size, input_dim)), key]
    params = model.init(key, x)['params']
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    data_iterator = get_data_iterator(dataset_path, batch_size)
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for batch in data_iterator:
            batch = batch['x'] / 255.
            losses, state = train_step(state, batch, kl_weight)
            loss, bce_loss, kld_loss = losses
            losses_this_epoch.append(bce_loss)
            print('BCE Loss:', bce_loss, 'KLD Loss:', kld_loss, 'Total Loss:', loss)
        print(
            'Epoch:', epoch, 
            'BCE Loss:', sum(losses_this_epoch)/len(losses_this_epoch)
        )
        test_latents = jax.random.normal(
            jax.random.PRNGKey(state.step), 
            (num_generative_samples, latent_dim)
        )
        test_generations = decoder.apply({'params': state.params['decoder']}, test_latents)
        test_generations = nn.sigmoid(test_generations)
        test_generations = jnp.reshape(
            test_generations, 
            (num_generative_samples, image_height, image_width)
        )
        print(test_generations.shape)
        for i in range(test_generations.shape[0]):
            plt.imsave(
                f'data/vae_reconstructions/image{i}_epoch{epoch}.png', 
                test_generations[i], cmap='gray'
            )

def train_transformer():
    wandb.init(project='transformer-vae')
    learning_rate = 1e-4
    num_epochs = 20
    dataset_path = 'data/easy-mnist/mnist_numpy_flat/data'
    batch_size = 32
    num_attention_heads = 1
    original_length = 784
    token_dim = 32
    x = jnp.ones([3, original_length])
    x = tokenize_batch(token_dim, x)
    print(x.shape)
    context_length = x.shape[-2]
    encoder = Encoder(
        attention_dim=token_dim,
        attention_out_dim=16,
        context_length=context_length,
        num_attention_heads=num_attention_heads,
        feed_forward_dim=8,
        final_dim=2
    )
    decoder = Decoder(
        attention_dim=token_dim,
        attention_out_dim=token_dim,
        context_length=context_length,
        num_attention_heads=num_attention_heads,
        feed_forward_dim=token_dim,
        final_dim=token_dim
    )
    model = Vae(encoder, decoder)
    key = jax.random.PRNGKey(0)
    params = model.init(key, [x, key])['params']
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    benchmark_sample = jnp.load(os.path.join(dataset_path, '0.npy'))
    benchmark_sample = jnp.expand_dims(benchmark_sample, axis=0)
    benchmark_sample = tokenize_batch(token_dim, benchmark_sample)
    benchmark_key = jax.random.PRNGKey(0)

    data_iterator = get_data_iterator(dataset_path, batch_size)
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for batch in data_iterator:
            batch = tokenize_batch(token_dim, batch['x']) / 255.
            loss, state = train_step(state, batch)
            losses_this_epoch.append(loss)
            print(loss)
        print(f'Finished epoch {epoch}')
        wandb.log({'loss': sum(losses_this_epoch)/len(losses_this_epoch)}, step=state.step)
        benchmark_reconstruction, _, _ = state.apply_fn(
            {'params': state.params}, [benchmark_sample, benchmark_key]
        )
        benchmark_reconstruction = detokenize_batch(original_length, benchmark_reconstruction)
        benchmark_reconstruction = jnp.reshape(benchmark_reconstruction, (28, 28))
        plt.imsave(
            f'data/vae_reconstructions/transformer_{epoch}.png', 
            benchmark_reconstruction, cmap='gray'
        )

def main():
    #train_transformer()
    train_baseline()
    #test_this()

if __name__ == '__main__':
    main()