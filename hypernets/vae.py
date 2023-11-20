import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
from hypernets.common.nn import LinearAttention
import jax

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
        print('e', e.shape)
        print('xb', x.shape)
        x = x + e
        print('x', x.shape)
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
    
class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, x):
        x, key = x
        means, deviations = self.encoder(x)
        print('means', means.shape)
        print('deviations', deviations.shape)
        x = jax.vmap(
            jax.vmap(sample_normals, in_axes=(0, 0, None)), in_axes=(0, 0, None)
        )(means, deviations, key)
        x = self.decoder(x)
        return x, means, deviations

def main():
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
    model = VAE(encoder, decoder)
    key = jax.random.PRNGKey(0)
    print(model.tabulate(key, [x, key]))

if __name__ == '__main__':
    main()