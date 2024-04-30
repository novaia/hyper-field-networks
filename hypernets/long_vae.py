# VAE for long sequences (i.e. neural fields with a lot of parameters).
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
import math

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
    device = str(jax.devices('gpu')[0])
    train = train.with_format('jax', device=device)
    test = test.with_format('jax', device=device)
    context_length = train[0]['params'].shape[0]
    return train, test, field_config, param_map, context_length

class LongVae(nn.Module):
    original_context_length: int
    context_length: int
    input_dim: int
    internal_dim: int
    latent_dim: int
    attention_dim: int
    num_attention_heads: int
    num_encoder_blocks: int
    num_decoder_blocks: int
    feed_forward_depth: int
    dropout_rate: float
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train):
        x = jax.vmap(lambda x: jnp.resize(x, (self.context_length, self.input_dim)))(x)
        positions = jnp.arange(self.context_length, dtype=jnp.uint32)
        pos_emb = nn.Embed(
            num_embeddings=self.context_length, features=self.internal_dim
        )(positions)
        x = nn.Dense(features=self.internal_dim)(x)
        x = x + pos_emb
        
        for _ in range(self.feed_forward_depth):
            x = nn.Dense(features=self.internal_dim)(x)
            x = nn.gelu(x)

        for _ in range(self.num_encoder_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.attention_dim*self.num_attention_heads,
                out_features=self.internal_dim,
                dtype=self.dtype
            )(inputs_q=x)
            x = x + residual
            x = nn.LayerNorm()(x)
            for _ in range(self.feed_forward_depth):
                x = nn.Dense(features=self.internal_dim)(x)
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                x = nn.gelu(x)
            x = x + residual

        # Latent.
        z = nn.Dense(features=self.latent_dim)(x)
        x = z

        for _ in range(self.feed_forward_depth):
            x = nn.Dense(features=self.internal_dim)(x)
            x = nn.gelu(x)

        for _ in range(self.num_decoder_blocks):
            residual = x
            x = nn.LayerNorm()(x)
            x = nn.MultiHeadDotProductAttention(
                num_heads=self.num_attention_heads,
                qkv_features=self.attention_dim*self.num_attention_heads,
                out_features=self.internal_dim,
                dtype=self.dtype
            )(inputs_q=x)
            x = x + residual
            x = nn.LayerNorm()(x)
            for _ in range(self.feed_forward_depth):
                x = nn.Dense(features=self.internal_dim)(x)
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
                x = nn.gelu(x)
            x = x + residual

        x = nn.Dense(features=self.input_dim)(x)
        x = jax.vmap(lambda x: jnp.ravel(x)[:self.original_context_length])(x)
        return x

@jax.jit
def train_step(state, x):
    key = jax.random.PRNGKey(state.step)
    def loss_fn(params):
        y = state.apply_fn(
            {'params': params}, 
            x=x, 
            train=True, 
            rngs={'dropout': key}
        )
        mse_loss = jnp.mean((x - y)**2)
        loss = mse_loss
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state.params)
    state = state.apply_gradients(grads=grad)
    return loss, state

@jax.jit
def test_step(state, x):
    y = state.apply_fn({'params': state.params}, x=x, train=False)
    mse_loss = jnp.mean((x - y)**2)
    return mse_loss

def main():
    output_path = 'data/long_vae_output/0'
    dataset_path = 'data/colored-monsters-ngp-image-18k'
    split_size = 0.1
    split_seed = 0
    train_set, test_set, field_config, param_map, original_context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_dim = 1000
    context_length = math.ceil(original_context_length / input_dim) 
    
    print('original_context_length', original_context_length)
    print('context_length', context_length)
    print('input_dim', input_dim)
    
    internal_dim = 1024
    latent_dim = 768
    attention_dim = 64
    num_attention_heads = 8
    num_encoder_blocks = 12
    num_decoder_blocks = 12
    feed_forward_depth = 4
    learning_rate = 1e-4
    dropout_rate = 0.1
    batch_size = 16
    num_epochs = 20

    model = LongVae(
        original_context_length=original_context_length,
        context_length=context_length,
        input_dim=input_dim,
        internal_dim=internal_dim,
        latent_dim=latent_dim,
        attention_dim=attention_dim,
        num_attention_heads=num_attention_heads,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        feed_forward_depth=4,
        dropout_rate=dropout_rate,
        dtype=jnp.bfloat16
    )
    
    x = jnp.ones((batch_size, original_context_length), dtype=jnp.float32)
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x=x, train=False)['params']
    opt = optax.adam(learning_rate=learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)

    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('num_train_samples', num_train_samples)
    print('train_steps', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('num_test_samples', num_test_samples)
    print('test_steps', test_steps)
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    test_sample = jnp.expand_dims(train_set[0]['params'], axis=0)
    print('test sample shape', test_sample.shape)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            batch = next(train_iterator)['params']
            loss, state = train_step(state, batch)
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            batch = next(test_iterator)['params']
            loss = test_step(state, batch)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print(f'epoch {epoch}, test_loss {average_test_loss}')

        flat_params = state.apply_fn({'params': state.params}, x=test_sample, train=False)[0]
        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_state = field_state.replace(params=params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{epoch}.png'), field_render)

if __name__ == '__main__':
    main()
