# 1D convolutional autoencoder.
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.99'

import jax
from jax import numpy as jnp
from flax import linen as nn
import optax
from orbax import checkpoint as ocp
from flax.training.train_state import TrainState
import datasets
import json
from typing import Any
from functools import partial
from hypernets.common.nn import SinusoidalEmbedding, kl_divergence
from float_tokenization import detokenize
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt
import math
import wandb

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

class EvenBetterConvAutoencoder(nn.Module):
    num_gn_groups: int
    per_channel_latent_dim: int
    dropout_rate: float
    hidden_features: list
    block_depth: int
    kernel_dim: int
    dtype: Any

    @nn.compact
    def __call__(self, x, train):
        def get_num_groups(num_features):
            if num_features >= self.num_gn_groups:
                return self.num_gn_groups
            else:
                return num_features

        # Encoder.
        for num_features in self.hidden_features:
            for _ in range(self.block_depth):
                x = nn.Conv(
                    features=num_features, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.Conv(
                    features=num_features, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.GroupNorm(num_groups=get_num_groups(num_features), dtype=self.dtype)(x)
            x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        pre_latent_dim = x.shape[1]
        x = jnp.swapaxes(x, 1, 2)
        x = nn.Dense(features=self.per_channel_latent_dim, dtype=self.dtype)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.gelu(x)
        x = nn.Dense(features=self.per_channel_latent_dim, dtype=self.dtype)(x)
        
        # Decoder.
        x = nn.Dense(features=self.per_channel_latent_dim, dtype=self.dtype)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.gelu(x)
        x = nn.Dense(features=pre_latent_dim, dtype=self.dtype)(x)
        x = jnp.swapaxes(x, 1, 2)
        for num_features in reversed(self.hidden_features):
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]), method='linear')
            for _ in range(self.block_depth):
                x = nn.Conv(
                    features=num_features, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.Conv(
                    features=num_features, kernel_size=(self.kernel_dim,), 
                    strides=(1,), padding='SAME', dtype=self.dtype
                )(x)
                x = nn.gelu(x)
                x = nn.GroupNorm(num_groups=get_num_groups(num_features), dtype=self.dtype)(x)

        x = nn.Conv(
            features=1, kernel_size=(self.kernel_dim,), 
            strides=(1,), padding='SAME', dtype=self.dtype       
        )(x)
        return x

def add_padding(x, left_padding, right_padding):
    left_zeros = jnp.zeros((x.shape[0], left_padding))
    right_zeros = jnp.zeros((x.shape[0], right_padding))
    x = jnp.concatenate([left_zeros, x, right_zeros], axis=-1)
    return x

def remove_padding(x, left_padding, right_padding):
    return x[:, left_padding:-right_padding, :]

def preprocess(x, left_padding, right_padding):
    x = add_padding(x, left_padding, right_padding)
    return jnp.expand_dims(x, axis=-1)

@partial(jax.jit, static_argnames=('left_padding', 'right_padding'))
def train_step(state, x, left_padding, right_padding):
    dropout_key = jax.random.PRNGKey(state.step)
    x = preprocess(x, left_padding, right_padding)
    
    def loss_fn(params):
        x_hat = state.apply_fn(
            {'params': params}, 
            x=x, 
            train=True, 
            rngs={'dropout': dropout_key}
        )
        return jnp.mean((x_hat - x)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

@partial(jax.jit, static_argnames=('left_padding', 'right_padding'))
def test_step(state, x, left_padding, right_padding):
    x = preprocess(x, left_padding, right_padding)
    x_hat = state.apply_fn({'params': state.params}, x, train=False)
    return jnp.mean((x_hat - x)**2)

@partial(jax.jit, static_argnames=('left_padding', 'right_padding'))
def reconstruct(state, x, left_padding, right_padding):
    x = preprocess(x, left_padding, right_padding)
    x_hat = state.apply_fn({'params': state.params}, x, train=False)
    return remove_padding(x_hat, left_padding, right_padding)

def main():
    checkpoint_path = None
    experiment_number = 0
    output_path = f'data/even_better_conv_ae_output/{experiment_number}/images'
    checkpoint_output_path = f'data/even_better_conv_ae_output/{experiment_number}/checkpoints'
    dataset_path = 'data/colored-monsters-ngp-image-18k'
    split_size = 0.1
    split_seed = 0
    train_set, test_set, field_config, param_map, context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    print('context_length', context_length)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_output_path):
        os.makedirs(checkpoint_output_path)

    padded_context_length = 18_000
    required_padding = padded_context_length - context_length
    left_padding = required_padding // 2
    if required_padding % 2 == 0:
        right_padding = left_padding
    else:
        right_padding = left_padding + 1
    print('required_padding', required_padding)
    print('left_padding', left_padding)
    print('right_padding', right_padding)

    num_epochs = 100
    batch_size = 16
    num_gn_groups = 16
    hidden_features = [16, 32, 64, 128]
    per_channel_latent_dim = 128
    dropout_rate = 0.3
    block_depth = 10
    kernel_dim = 5
    learning_rate = 3e-4
    weight_decay = 1e-4

    wandb_config = {
        'context_length': context_length,
        'padded_context_length': padded_context_length,
        'left_padding': left_padding,
        'right_padding': right_padding,
        'batch_size': batch_size,
        'num_gn_groups': num_gn_groups,
        'per_channel_latent_dim': per_channel_latent_dim,
        'dropout_rate': dropout_rate,
        'hidden_features': hidden_features,
        'block_depth': block_depth,
        'kernel_dim': kernel_dim,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    model = EvenBetterConvAutoencoder(
        num_gn_groups=num_gn_groups,
        per_channel_latent_dim=per_channel_latent_dim,
        dropout_rate=dropout_rate,
        hidden_features=hidden_features,
        block_depth=block_depth,
        kernel_dim=kernel_dim,
        dtype=jnp.bfloat16
    )
    x = preprocess(
        x=jnp.ones((batch_size, context_length), dtype=jnp.float32),
        left_padding=left_padding,
        right_padding=right_padding
    )
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x=x, train=False)['params']
    opt = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    wandb_config['param_count'] = param_count
    print(f'param_count: {param_count:,}')
    
    num_train_samples = len(train_set)
    train_steps = num_train_samples // batch_size
    print('num_train_samples', num_train_samples)
    print('train_steps', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // batch_size
    print('num_test_samples', num_test_samples)
    print('test_steps', test_steps)
    
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if checkpoint_path is not None:
        print(f'loading checkpoint {checkpoint_path}')
        checkpointer.restore(os.path.abspath(checkpoint_path), item=state)

    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    wandb.init(project='conv-vae', config=wandb_config)
    test_sample = jnp.expand_dims(test_set[0]['params'], axis=0)
    print('test sample shape', test_sample.shape)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            batch = next(train_iterator)['params']
            loss, state = train_step(state, batch, left_padding, right_padding)
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        wandb.log({'loss': average_loss}, step=state.step)
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            batch = next(test_iterator)['params']
            loss = test_step(state, batch, left_padding, right_padding)
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print(f'epoch {epoch}, test_loss {average_test_loss}')
        wandb.log({'test_loss': average_test_loss}, step=state.step)
        current_checkpoint_path = os.path.join(
            os.path.abspath(checkpoint_output_path), f'step{state.step}'
        )
        checkpointer.save(current_checkpoint_path, state, force=True)
        print(f'saved checkpoint {current_checkpoint_path}')
        flat_params = jnp.squeeze(reconstruct(state, test_sample, left_padding, right_padding)[0], axis=-1)
        params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
        field_state = field_state.replace(params=params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(output_path, f'{state.step}.png'), field_render)

if __name__ == '__main__':
    main()
