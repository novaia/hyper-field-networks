# The purpose of this script is to train 2 separate autoencoders on the 
# MLP and hash grid sections of NGP image fields respectively.
# The motiviation for this is the large diifference in parameter scales
# between the hash grid and MLP sections (check ../scripts/min_max_scatter_plot.py).
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
    latent_features: int
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
        for i, num_features in enumerate(self.hidden_features):
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
            if i != len(self.hidden_features) - 1:
                x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        x = nn.Conv(
            features=self.latent_features, kernel_size=(self.kernel_dim,), 
            strides=(1,), padding='SAME', dtype=self.dtype
        )(x)

        # Decoder.
        for i, num_features in enumerate(list(reversed(self.hidden_features))):
            if i != 0:
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

def add_padding(x, left_padding, right_padding, requires_padding):
    if not requires_padding:
        return x
    else:
        left_zeros = jnp.zeros((x.shape[0], left_padding), dtype=x.dtype)
        right_zeros = jnp.zeros((x.shape[0], right_padding), dtype=x.dtype)
        x = jnp.concatenate([left_zeros, x, right_zeros], axis=-1)
        return x

def remove_padding(x, left_padding, right_padding, requires_padding):
    if not requires_padding:
        return x
    else:
        x = x[..., left_padding:]
        x = x[..., :-right_padding]
        return x

def preprocess(
    x, train_on_hash_grid, hash_grid_end, 
    left_padding, right_padding, requires_padding
):
    if train_on_hash_grid:
        # Isolate hash grid section.
        x = x[..., :hash_grid_end]
    else:
        # Isolate MLP section.
        x = x[..., hash_grid_end:]
    x = add_padding(x, left_padding, right_padding, requires_padding)
    return jnp.expand_dims(x, axis=-1)

@partial(jax.jit, static_argnames=(
    'train_on_hash_grid', 'hash_grid_end', 
    'left_padding', 'right_padding', 'requires_padding'
))
def train_step(
    state, batch, train_on_hash_grid, hash_grid_end,
    left_padding, right_padding, requires_padding
):
    x_in = preprocess(
        batch, train_on_hash_grid, hash_grid_end, 
        left_padding, right_padding, requires_padding
    )
    
    def loss_fn(params):
        x_out = state.apply_fn({'params': params}, x=x_in, train=True)
        return jnp.mean((x_out - x_in)**2)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

@partial(jax.jit, static_argnames=(
    'train_on_hash_grid', 'hash_grid_end', 
    'left_padding', 'right_padding', 'requires_padding'
))
def test_step(
    state, batch, train_on_hash_grid, hash_grid_end, 
    left_padding, right_padding, requires_padding
):
    x_in = preprocess(
        batch, train_on_hash_grid, hash_grid_end,
        left_padding, right_padding, requires_padding
    )
    x_out = state.apply_fn({'params': state.params}, x=x_in, train=False)
    return jnp.mean((x_out - x_in)**2)

@partial(jax.jit, static_argnames=(
    'train_on_hash_grid', 'hash_grid_end', 
    'left_padding', 'right_padding', 'requires_padding'
))
def reconstruct(
    state, batch, train_on_hash_grid, hash_grid_end,
    left_padding, right_padding, requires_padding
):
    # For reconstruction we only reconstruct the section that the autoencoder is being
    # trained on. After reconstruction we concatenate it with the other section so the 
    # field can be rendered.
    if train_on_hash_grid:
        x_in = batch[..., :hash_grid_end] # Hash grid section.
        other_section = batch[..., hash_grid_end:] # MLP section.
    else:
        x_in = batch[..., hash_grid_end:] # MLP section.
        other_section = batch[..., :hash_grid_end] # Hash grid section.

    x_in = add_padding(x_in, left_padding, right_padding, requires_padding)
    x_in = jnp.expand_dims(x_in, axis=-1)
    x_out = state.apply_fn({'params': state.params}, x=x_in, train=False)
    x_out = jnp.squeeze(x_out, axis=-1)
    x_out = remove_padding(x_out, left_padding, right_padding, requires_padding)
    
    # Hash grid section goes first, MLP section goes last.
    if train_on_hash_grid:
        reconstruction = jnp.concatenate([x_out, other_section], axis=-1)
    else:
        reconstruction = jnp.concatenate([other_section, x_out], axis=-1)
    return reconstruction

def calculate_required_padding(sequence_length, num_downsamples):
    sequence_length = int(sequence_length)
    num_downsamples = int(num_downsamples)
    required_division = 2**num_downsamples
    rounded_quotient = math.ceil(sequence_length / required_division)
    padded_sequence_length = int(rounded_quotient * required_division)
    
    if padded_sequence_length == sequence_length:
        left_padding = 0
        right_padding = 0
        requires_padding = False
        return left_padding, right_padding, requires_padding
    else:
        requires_padding = True
        total_padding = padded_sequence_length - sequence_length
        left_padding = total_padding // 2
        if total_padding % 2 == 0:
            right_padding = left_padding
        else:
            right_padding = left_padding + 1
        return left_padding, right_padding, requires_padding

def main():
    checkpoint_path = None
    experiment_number = 3
    output_path = f'data/split_field_conv_ae_output/{experiment_number}/images'
    checkpoint_output_path = f'data/split_field_conv_ae_output/{experiment_number}/checkpoints'
    dataset_path = 'data/colored-monsters-ngp-image-18k'
    split_size = 0.1
    split_seed = 0
    train_set, test_set, field_config, param_map, context_length = \
        load_dataset(dataset_path, split_size, split_seed)
    print('context_length', context_length)
    hash_grid_end = (
        field_config['num_hash_table_levels'] 
        * field_config['max_hash_table_entries'] 
        * field_config['hash_table_feature_dim']
    )
    print('hash_grid_end', hash_grid_end)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_output_path):
        os.makedirs(checkpoint_output_path)
    
    train_on_hash_grid = False 
    num_epochs = 100
    batch_size = 32
    num_gn_groups = 32
    hidden_features = [16, 32, 64, 64]
    latent_features = 8
    block_depth = 4
    kernel_dim = 5
    learning_rate = 1e-4
    weight_decay = 1e-4

    if train_on_hash_grid:
        section_length = hash_grid_end 
        print('training on hash grid section...')
    else:
        section_length = context_length - hash_grid_end
        print('training on MLP section...')
    print('section_length', section_length)
    
    left_padding, right_padding, requires_padding = calculate_required_padding(
        sequence_length=section_length, num_downsamples=len(hidden_features)-1
    )
    print('requires_padding', requires_padding)
    print('left_padding', left_padding)
    print('right_padding', right_padding)

    wandb_config = {
        'train_on_hash_grid': train_on_hash_grid,
        'context_length': context_length,
        'section_length': section_length,
        'requires_padding': requires_padding,
        'left_padding': left_padding,
        'right_padding': right_padding,
        'batch_size': batch_size,
        'num_gn_groups': num_gn_groups,
        'latent_features': latent_features,
        'hidden_features': hidden_features,
        'block_depth': block_depth,
        'kernel_dim': kernel_dim,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }

    model = EvenBetterConvAutoencoder(
        num_gn_groups=num_gn_groups,
        latent_features=latent_features,
        hidden_features=hidden_features,
        block_depth=block_depth,
        kernel_dim=kernel_dim,
        dtype=jnp.bfloat16
    )

    x = preprocess(
        x=jnp.ones((batch_size, context_length), dtype=jnp.float32),
        train_on_hash_grid=train_on_hash_grid,
        hash_grid_end=hash_grid_end,
        left_padding=left_padding,
        right_padding=right_padding,
        requires_padding=requires_padding
    )
    params_key = jax.random.PRNGKey(91)
    params = model.init(params_key, x=x, train=False)['params']
    opt = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    wandb_config['param_count'] = param_count
    print(f'param_count {param_count:,}')
    
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
    num_preview_samples = 5
    print('num_preview_samples', num_preview_samples)
    preview_samples = test_set[:num_preview_samples]['params']
    print('preview_samples shape', preview_samples.shape)
    for epoch in range(num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            batch = next(train_iterator)['params']
            loss, state = train_step(
                state=state, batch=batch, train_on_hash_grid=train_on_hash_grid, 
                hash_grid_end=hash_grid_end, left_padding=left_padding,
                right_padding=right_padding, requires_padding=requires_padding
            )
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        wandb.log({'loss': average_loss}, step=state.step)
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(batch_size)
        losses_this_test = []
        for step in range(test_steps):
            batch = next(test_iterator)['params']
            loss = test_step(
                state=state, batch=batch, train_on_hash_grid=train_on_hash_grid, 
                hash_grid_end=hash_grid_end, left_padding=left_padding, 
                right_padding=right_padding, requires_padding=requires_padding
            )
            losses_this_test.append(loss)
        average_test_loss = sum(losses_this_test) / len(losses_this_test)
        print(f'epoch {epoch}, test_loss {average_test_loss}')
        wandb.log({'test_loss': average_test_loss}, step=state.step)
        current_checkpoint_path = os.path.join(
            os.path.abspath(checkpoint_output_path), f'step{state.step}'
        )
        checkpointer.save(current_checkpoint_path, state, force=True)
        print(f'saved checkpoint {current_checkpoint_path}')
        reconstructed_preview_samples = reconstruct(
            state=state, batch=preview_samples, train_on_hash_grid=train_on_hash_grid, 
            hash_grid_end=hash_grid_end, left_padding=left_padding, 
            right_padding=right_padding, requires_padding=requires_padding
        )
        for i in range(num_preview_samples):
            flat_params = reconstructed_preview_samples[i]
            params = unflatten_params(jnp.array(flat_params, dtype=jnp.float32), param_map)
            field_state = field_state.replace(params=params)
            field_render = ngp_image.render_image(
                field_state, field_config['image_height'], 
                field_config['image_width'], field_config['channels']
            )
            field_render = jax.device_put(field_render, jax.devices('cpu')[0])
            plt.imsave(os.path.join(output_path, f'{state.step}_{i}.png'), field_render)

if __name__ == '__main__':
    main()
