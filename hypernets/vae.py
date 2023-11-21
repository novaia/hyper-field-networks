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

class AutoPositionalEmbedding(nn.Module):
    num_positions: int
    feature_dim: int

    @nn.compact
    def __call__(self):
        positions = jnp.arange(self.num_positions)
        e = nn.Embed(self.num_positions, self.feature_dim)(positions)
        return e

class TransformerVaeComponent(nn.Module):
    hidden_dims: List[int]
    output_dim: int
    per_head_attention_dims: List[int]
    num_attention_heads: int
    context_length: int
    encoder: bool

    @nn.compact
    def __call__(self, x):
        e = AutoPositionalEmbedding(self.context_length, x.shape[-1])()
        x = x + e
        for i in range(len(self.hidden_dims)):
            current_hidden_dim = self.hidden_dims[i]
            x = LinearAttention(
                attention_dim=self.per_head_attention_dims[i] * self.num_attention_heads, 
                output_dim=current_hidden_dim,
                num_heads=self.num_attention_heads
            )(x)
            x = nn.Dense(features=current_hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=current_hidden_dim)(x)
            x = nn.gelu(x)
        if self.encoder:
            means = nn.Dense(features=self.output_dim)(x)
            deviations = nn.Dense(features=self.output_dim)(x)
            deviations = jnp.exp(deviations)
            return means, deviations
        else:
            logits = nn.Dense(features=self.output_dim)(x)
            return logits

class TransformerVae(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x):
        x, key = x
        means, deviations = self.encoder(x)
        x = means + deviations * jax.random.normal(key, means.shape)
        logits = self.decoder(x)
        return logits, means, deviations
    
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
            (loss, bce_loss, kld_loss), state = train_step(state, batch, kl_weight)
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
    kl_weight = 0.0
    learning_rate = 1e-4
    num_epochs = 20
    dataset_path = 'data/easy-mnist/mnist_numpy_flat/data'
    batch_size = 32
    original_length = 784
    token_dim = 784//2
    x = jnp.ones([3, original_length])
    x = tokenize_batch(token_dim, x)
    print(x.shape)
    context_length = x.shape[-2]
    
    latent_dim = 2
    num_attention_heads = 4
    encoder_attention_dims = [128, 128]
    decoder_attention_dims = list(reversed(encoder_attention_dims))
    encoder_hidden_dims = [512, 512]
    decoder_hidden_dims = list(reversed(encoder_hidden_dims))
    encoder = TransformerVaeComponent(
        hidden_dims=encoder_hidden_dims,
        output_dim=latent_dim,
        per_head_attention_dims=encoder_attention_dims,
        num_attention_heads=num_attention_heads,
        context_length=context_length,
        encoder=True
    )
    decoder = TransformerVaeComponent(
        hidden_dims=decoder_hidden_dims,
        output_dim=token_dim,
        per_head_attention_dims=decoder_attention_dims,
        num_attention_heads=num_attention_heads,
        context_length=context_length,
        encoder=False
    )

    model = TransformerVae(encoder, decoder)
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
            (loss, bce_loss, kld_loss), state = train_step(state, batch, kl_weight)
            losses_this_epoch.append(bce_loss)
            print(bce_loss)
        print(f'Finished epoch {epoch}')
        benchmark_reconstruction, _, _ = state.apply_fn(
            {'params': state.params}, [benchmark_sample, benchmark_key]
        )
        benchmark_reconstruction = detokenize_batch(original_length, benchmark_reconstruction)
        benchmark_reconstruction = nn.sigmoid(benchmark_reconstruction)
        benchmark_reconstruction = jnp.reshape(benchmark_reconstruction, (28, 28))
        plt.imsave(
            f'data/vae_reconstructions/transformer_{epoch}.png', 
            benchmark_reconstruction, cmap='gray'
        )

def main():
    train_transformer()
    #train_baseline()
    #test_this()

if __name__ == '__main__':
    main()