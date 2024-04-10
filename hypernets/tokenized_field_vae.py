import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from flax.training.train_state import TrainState
import datasets
import os, json
from typing import Any
from functools import partial
from hypernets.common.nn import SinusoidalEmbedding, kl_divergence
from float_tokenization import detokenize
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt

def load_dataset(dataset_path, test_size, split_seed):
    field_config = None
    with open(os.path.join(dataset_path, 'field_config.json'), 'r') as f:
        field_config = json.load(f)
    assert field_config is not None
    param_map = None
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        param_map = json.load(f)
    assert param_map is not None
    
    parquet_dir = os.path.join(dataset_path, 'data')
    parquet_paths = [
        os.path.join(parquet_dir, p) 
        for p in os.listdir(parquet_dir) if p.endswith('.parquet')
    ]
    num_parquet_files = len(parquet_paths)
    assert num_parquet_files > 0
    print(f'Found {num_parquet_files} parquet file(s) in dataset directory')

    dataset = datasets.load_dataset('parquet', data_files=parquet_paths)
    train, test = dataset['train'].train_test_split(test_size=test_size, seed=split_seed).values()
    return train.with_format('jax'), test.with_format('jax'), field_config, param_map

class TokenizedFieldVae(nn.Module):
    vocab_size: int
    context_length: int
    embedding_dim: int
    latent_dim: int
    num_attention_heads: int
    num_encoder_blocks: int
    num_decoder_blocks: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, tokens, train, key):
        dropout_rate = 0.4
        def transformer_block(x, num_blocks):
            for _ in range(num_blocks):
                residual = x
                x = nn.LayerNorm(dtype=self.dtype)(x)
                x = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_attention_heads,
                    qkv_features=self.embedding_dim,
                    out_features=self.embedding_dim,
                    dtype=self.dtype
                )(inputs_q=x)
                x = x + residual
                residual = x
                x = nn.LayerNorm(dtype=self.dtype)(x)
                x = nn.Dropout(rate=dropout_rate, broadcast_dims=(-1, -2))(x, deterministic=not train, rng=key)
                x = nn.Dense(features=self.embedding_dim, dtype=self.dtype)(x)
                x = nn.gelu(x)
                x = nn.Dropout(rate=dropout_rate, broadcast_dims=(-1, -2))(x, deterministic=not train, rng=key)
                x = nn.Dense(features=self.embedding_dim, dtype=self.dtype)(x)
                x = nn.gelu(x)
                x = x + residual
                return x

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embedding_dim)(tokens)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(num_embeddings=self.context_length, features=self.embedding_dim)(positions)
        x = x + pos_emb
        x = transformer_block(x, self.num_encoder_blocks)
        #means = nn.Dense(features=self.latent_dim, kernel_init=nn.initializers.zeros_init())(x)
        #stds = jnp.exp(nn.Dense(features=self.latent_dim, kernel_init=nn.initializers.zeros_init())(x))
        x = nn.Dense(features=self.latent_dim)(x)
        #x = means + stds * jax.random.normal(key, means.shape)
        x = nn.Dense(features=self.embedding_dim)(x)
        x = transformer_block(x, self.num_decoder_blocks)
        logits = nn.Dense(features=self.vocab_size)(x)
        return logits#, means, stds

def make_kl_schedule(initial_value, final_value, transition_steps, cycle_steps):
    slope = final_value / transition_steps
    linear_fn = partial(
        lambda m, b, max_y, x: jnp.clip(m * x + b, 0, max_y), 
        slope, initial_value, final_value
    )
    cyclical_fn = partial(lambda period, x: linear_fn(x % period), cycle_steps)
    return cyclical_fn

@jax.jit
def train_step(state, tokens, kl_weight):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, tokens, True, key)
        ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
        #kld_loss = jnp.mean(stds**2 + means**2 - jnp.log(stds) - 0.5)
        kld_loss = 0.0
        #loss = ce_loss + (kld_loss * kl_weight)
        loss = ce_loss
        return loss, (ce_loss, kld_loss)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (ce_loss, kld_loss)), grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return (loss, ce_loss, kld_loss), state

@jax.jit
def test_step(state, tokens, seed):
    key = jax.random.PRNGKey(seed)
    logits = state.apply_fn({'params': state.params}, tokens, False, key)
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens))
    #kld_loss = jnp.mean(stds**2 + means**2 - jnp.log(stds) - 0.5)
    kld_loss = 0.0
    return ce_loss, kld_loss

def main():
    output_path = 'data/tokenized_field_vae_output/22'
    dataset_path = 'data/mnist-ngp-image-612-11bit'
    split_size = 0.1
    split_seed = 0
    train_set, test_set, field_config, param_map = load_dataset(dataset_path, split_size, split_seed)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    token_bits = 11
    vocab_size = 2 * 2**token_bits - 1
    print('vocab size', vocab_size)

    num_epochs = 200
    batch_size = 80
    context_length = 612
    embedding_dim = 512
    latent_dim = 8
    num_attention_heads = 16
    num_encoder_blocks = 12
    num_decoder_blocks = 12
    learning_rate = 1e-4
    weight_decay = 1e-2

    model = TokenizedFieldVae(
        vocab_size=vocab_size,
        context_length=context_length,
        embedding_dim=embedding_dim,
        latent_dim=latent_dim,
        num_attention_heads=num_attention_heads,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        dtype=jnp.bfloat16
    )
    x = jnp.ones((batch_size, context_length), dtype=jnp.uint32)
    vae_key = jax.random.PRNGKey(68)
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x, True, vae_key)['params']
    opt = optax.chain(
        optax.zero_nans(),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('test set size', num_train_samples)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('train set size', num_test_samples)
    
    cycle_steps = train_steps 
    kl_weight_schedule = make_kl_schedule(
        initial_value=0.0,
        final_value=0.6,
        transition_steps=cycle_steps,
        cycle_steps=cycle_steps
    )
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    test_sample = jnp.expand_dims(test_set[0]['tokens'], axis=0)
    print('test sample shape', test_sample.shape)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        ce_losses_this_epoch = []
        kld_losses_this_epoch = []
        for step in range(train_steps):
            tokens = next(train_iterator)['tokens']
            kl_weight = kl_weight_schedule(state.step)
            (loss, ce_loss, kld_loss), state = train_step(state, tokens, kl_weight)
            losses_this_epoch.append(loss)
            ce_losses_this_epoch.append(ce_loss)
            kld_losses_this_epoch.append(kld_loss)
            #print(f'step {step}, loss {loss}, ce_loss {ce_loss}, kld_loss {kld_loss}')
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        average_ce_loss = sum(ce_losses_this_epoch) / len(ce_losses_this_epoch)
        average_kld_loss = sum(kld_losses_this_epoch) / len(kld_losses_this_epoch)
        print(f'epoch {epoch}, ce_loss {average_ce_loss}, kld_loss {average_kld_loss}, total_loss {average_loss}')
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        ce_losses_this_test = []
        kld_losses_this_test = []
        for step in range(test_steps):
            tokens = next(test_iterator)['tokens']
            ce_loss, kld_loss = test_step(state, tokens, step)
            ce_losses_this_test.append(ce_loss)
            kld_losses_this_test.append(kld_loss)
        average_test_ce_loss = sum(ce_losses_this_test) / len(ce_losses_this_test)
        average_test_kld_loss = sum(kld_losses_this_test) / len(kld_losses_this_test)
        print(f'epoch {epoch}, test_ce_loss {average_test_ce_loss}, test_kld_loss {average_test_kld_loss}')

        logits = state.apply_fn({'params': state.params}, test_sample, False, jax.random.PRNGKey(0))
        probs = nn.softmax(logits[0])
        tokens = jax.vmap(lambda p: jnp.argmax(p), in_axes=0)(probs)
        flat_params = detokenize(tokens)
        # Detokenized NaNs should not exist so there is probably a bug in the detokenization code.
        flat_params = jnp.nan_to_num(flat_params)
        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_state = field_state.replace(params=params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), field_render)

if __name__ == '__main__':
    main()
