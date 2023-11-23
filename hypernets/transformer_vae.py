import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from typing import List
from hypernets.common.nn import \
    TransformerVae, TransformerVaeEncoder, TransformerVaeDecoder, \
    binary_cross_entropy_with_logits, kl_divergence
import optax
from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
import matplotlib.pyplot as plt
import os
from functools import partial

def create_transformer_vae(
    hidden_dims, latent_dim, output_dim, context_length, num_attention_heads
):
    encoder_hidden_dims = hidden_dims
    decoder_hidden_dims = list(reversed(encoder_hidden_dims))
    encoder = TransformerVaeEncoder(
        context_length=context_length,
        hidden_dims=encoder_hidden_dims,
        num_attention_heads=num_attention_heads,
        latent_dim=latent_dim
    )
    decoder = TransformerVaeDecoder(
        context_length=context_length,
        hidden_dims=decoder_hidden_dims,
        num_attention_heads=num_attention_heads,
        output_dim=output_dim
    )
    vae = TransformerVae(encoder, decoder)
    return vae, encoder, decoder

def tokenize_batch(token_dim, batch):
    context_length = int(jnp.ceil(batch.shape[1] / token_dim))
    batch = jnp.resize(batch, (batch.shape[0], context_length, token_dim))
    return batch

def detokenize_batch(original_dim, batch):
    batch = jnp.resize(batch, (batch.shape[0], original_dim))
    return batch

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

def cyclical_linear_schedule(initial_value, final_value, transition_steps, cycle_steps):
    slope = final_value / transition_steps
    linear_fn = partial(
        lambda m, b, max_y, x: jnp.clip(m * x + b, 0, max_y), 
        slope, initial_value, final_value
    )
    cyclical_fn = partial(lambda period, x: linear_fn(x % period), cycle_steps)
    return cyclical_fn

@jax.jit
def train_step(state, batch, kl_weight):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits, means, logvars = state.apply_fn({'params': params}, [batch, key])
        bce_loss = jnp.mean(binary_cross_entropy_with_logits(logits, batch))
        kld_loss = jnp.mean(kl_divergence(means, logvars))
        loss = bce_loss + (kld_loss * kl_weight)
        return loss, (bce_loss, kld_loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (bce_loss, kld_loss)), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return (loss, bce_loss, kld_loss), state

def main():
    key = jax.random.PRNGKey(0)
    
    image_height = 28
    image_width = 28
    num_generative_samples = 3

    original_dim = 784
    token_dim = 14
    x = jnp.ones([3, original_dim])
    x = tokenize_batch(token_dim, x)
    context_length = x.shape[-2]
    hidden_dims = [128, 32]
    latent_dim = 16
    num_attention_heads = 8
    learning_rate = 1e-3
    num_epochs = 20
    batch_size = 64
    dataset_path = 'data/easy-mnist/mnist_numpy_flat/data'

    model, encoder, decoder = create_transformer_vae(
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        output_dim=token_dim,
        context_length=context_length,
        num_attention_heads=num_attention_heads
    )
    params = model.init(key, [x, key])['params']
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    cycle_steps = 4*(60_000//batch_size)
    kl_weight_schedule = cyclical_linear_schedule(
        initial_value=0.0,
        final_value=0.5,
        transition_steps=cycle_steps//2,
        cycle_steps=cycle_steps
    )

    benchmark_sample = jnp.load(os.path.join(dataset_path, '0.npy'))
    benchmark_sample = jnp.expand_dims(benchmark_sample, axis=0)
    benchmark_sample = tokenize_batch(token_dim, benchmark_sample)

    data_iterator = get_data_iterator(dataset_path, batch_size)
    for epoch in range(num_epochs):
        losses_this_epoch = []
        for batch in data_iterator:
            batch = tokenize_batch(token_dim, batch['x']) / 255.
            kl_weight = kl_weight_schedule(state.step)
            (loss, bce_loss, kld_loss), state = train_step(state, batch, kl_weight)
            losses_this_epoch.append(bce_loss)
            loss_message = (
                'BCE Loss: {:>8.5f}   ' + 
                'KLD Loss: {:>8.5f}   ' +
                'Total Loss: {:>8.5f}   ' +
                'KL Weight: {:>8.5f}'
            )
            print(loss_message.format(bce_loss, kld_loss, loss, kl_weight))
        print(f'Finished epoch {epoch}')
        test_latents = jax.random.normal(
            jax.random.PRNGKey(epoch), 
            (num_generative_samples, latent_dim)
        )
        test_generations = decoder.apply({'params': state.params['decoder']}, test_latents)
        test_generations = nn.sigmoid(test_generations)
        test_generations = detokenize_batch(original_dim, test_generations)
        test_generations = jnp.reshape(
            test_generations, 
            (num_generative_samples, image_height, image_width)
        )
        for i in range(test_generations.shape[0]):
            plt.imsave(f'data/tvae/image{i}_epoch{epoch}.png', test_generations[i], cmap='gray')

if __name__ == '__main__':
    main()
