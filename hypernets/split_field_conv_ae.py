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
from typing import Any, List, Tuple
from functools import partial
from fields.common.flattening import unflatten_params
from fields import ngp_image
import matplotlib.pyplot as plt
import math
import wandb
from dataclasses import dataclass
import argparse

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

class Encoder(nn.Module):
    num_norm_groups: int
    latent_features: int
    intermediate_features: list
    block_depth: int
    kernel_dim: int
    dtype: Any
    
    @nn.compact
    def __call__(self, x):
        for i, num_features in enumerate(self.intermediate_features):
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
                x = nn.GroupNorm(num_groups=min(num_features, self.num_norm_groups), dtype=self.dtype)(x)
            if i != len(self.intermediate_features) - 1:
                x = nn.max_pool(x, window_shape=(2,), strides=(2,), padding='VALID')
        x = nn.Conv(
            features=self.latent_features, kernel_size=(self.kernel_dim,), 
            strides=(1,), padding='SAME', dtype=self.dtype
        )(x)
        return x

class Decoder(nn.Module):
    num_norm_groups: int
    intermediate_features: list
    block_depth: int
    kernel_dim: int
    dtype: Any
    
    @nn.compact
    def __call__(self, x):
        for i, num_features in enumerate(self.intermediate_features):
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
                x = nn.GroupNorm(num_groups=min(num_features, self.num_norm_groups), dtype=self.dtype)(x)
        x = nn.Conv(
            features=1, kernel_size=(self.kernel_dim,), 
            strides=(1,), padding='SAME', dtype=self.dtype       
        )(x)
        return x

class Autoencoder(nn.Module):
    encoder: nn.Module 
    decoder: nn.Module

    @nn.compact
    def __call__(self, x, train):
        return self.decoder(self.encoder(x))


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

@dataclass
class SplitFieldConvAeConfig:
    model_name: str
    train_on_hash_grid: bool
    num_epochs: int
    batch_size: int
    num_norm_groups: int
    encoder_intermediate_features: List[int]
    decoder_intermediate_features: List[int]
    latent_features: int
    block_depth: int
    kernel_dim: int
    learning_rate: float
    weight_decay: float
    requires_padding: bool
    left_padding: int
    right_padding: int
    num_field_params: int
    num_hash_grid_params: int
    model_seed: int
    split_seed: int
    test_split_size: float

    def __init__(self, config_dict) -> None:
        self.model_name = config_dict['model_name']
        self.train_on_hash_grid = config_dict['train_on_hash_grid']
        self.num_epochs = config_dict['num_epochs']
        self.batch_size = config_dict['batch_size']
        self.num_norm_groups = config_dict['num_norm_groups']
        self.encoder_intermediate_features = config_dict['intermediate_features'] 
        self.decoder_intermediate_features = list(reversed(self.encoder_intermediate_features))
        self.latent_features = config_dict['latent_features']
        self.block_depth = config_dict['block_depth'] 
        self.kernel_dim = config_dict['kernel_dim']
        self.learning_rate = config_dict['learning_rate'] 
        self.weight_decay = config_dict['weight_decay']
        self.requires_padding = config_dict['requires_padding']
        self.left_padding = config_dict['left_padding']
        self.right_padding = config_dict['right_padding']
        if not self.requires_padding:
            assert self.left_padding == 0 and self.right_padding == 0, (
                'Config specified requires_padding=False, but the left_padding and right_padding ',
                'values are not 0'
            )
        self.num_field_params = config_dict['num_field_params']
        self.num_hash_grid_params = config_dict['num_hash_grid_params']
        self.model_seed = config_dict['model_seed']
        self.split_seed = config_dict['split_seed']
        self.test_split_size = config_dict['test_split_size']

def init_model_from_config(
    model_config: SplitFieldConvAeConfig
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    encoder_model = Encoder(
        num_norm_groups=model_config.num_norm_groups,
        intermediate_features=model_config.encoder_intermediate_features,
        latent_features=model_config.latent_features,
        block_depth=model_config.block_depth,
        kernel_dim=model_config.kernel_dim,
        dtype=jnp.bfloat16
    )
    decoder_model = Decoder(
        num_norm_groups=model_config.num_norm_groups,
        intermediate_features=model_config.decoder_intermediate_features,
        block_depth=model_config.block_depth,
        kernel_dim=model_config.kernel_dim,
        dtype=jnp.bfloat16
    )
    autoencoder_model = Autoencoder(encoder=encoder_model, decoder=decoder_model)
    return autoencoder_model, encoder_model, decoder_model

def init_model_state(
    key: Any, model: nn.Module, 
    model_config: SplitFieldConvAeConfig, use_batch_size: bool = True
) -> TrainState:
    init_batch_size = model_config.batch_size if use_batch_size else 1
    x = preprocess(
        x=jnp.ones((init_batch_size, model_config.num_field_params), dtype=jnp.float32),
        train_on_hash_grid=model_config.train_on_hash_grid,
        hash_grid_end=model_config.num_hash_grid_params,
        left_padding=model_config.left_padding,
        right_padding=model_config.right_padding,
        requires_padding=model_config.requires_padding
    )
    params = model.init(key, x=x, train=False)['params']
    opt = optax.adamw(
        learning_rate=model_config.learning_rate, 
        weight_decay=model_config.weight_decay
    )
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    return state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calc-padding', action='store_true')
    args = parser.parse_args()
    
    config_path = 'configs/split_field_conv_ae_mlp.json'
    with open(config_path, 'r') as f:
        main_config_dict = json.load(f)
        main_config = SplitFieldConvAeConfig(main_config_dict)
    checkpoint_path = None
    experiment_number = 7
    output_path = f'data/split_field_conv_ae_output/{experiment_number}/images'
    checkpoint_output_path = f'data/split_field_conv_ae_output/{experiment_number}/checkpoints'
    dataset_path = 'data/colored-monsters-ngp-image-18k'
    train_set, test_set, field_config, param_map, context_length = load_dataset(
        dataset_path=dataset_path, 
        test_size=main_config.test_split_size, 
        split_seed=main_config.split_seed
    )
    print('context_length', main_config.num_field_params)
    #hash_grid_end = (
    #    field_config['num_hash_table_levels'] 
    #    * field_config['max_hash_table_entries'] 
    #    * field_config['hash_table_feature_dim']
    #)
    print('hash_grid_end', main_config.num_hash_grid_params)

    if main_config.train_on_hash_grid:
        section_length = main_config.num_hash_grid_params
        print('training on hash grid section...')
    else:
        section_length = main_config.num_field_params - main_config.num_hash_grid_params
        print('training on MLP section...')
    print('section_length', section_length)
    
    if args.calc_padding:
        print('Calculating padding configuration...')
        _left_padding, _right_padding, _requires_padding = calculate_required_padding(
            sequence_length=section_length, 
            num_downsamples=len(main_config.encoder_intermediate_features)-1
        )
        print('requires_padding:', _requires_padding)
        print('left_padding:', _left_padding)
        print('right_padding:', _right_padding)
        exit()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(checkpoint_output_path):
        os.makedirs(checkpoint_output_path)
    
    encoder_model = Encoder(
        num_norm_groups=main_config.num_norm_groups,
        intermediate_features=main_config.encoder_intermediate_features,
        latent_features=main_config.latent_features,
        block_depth=main_config.block_depth,
        kernel_dim=main_config.kernel_dim,
        dtype=jnp.bfloat16
    )
    decoder_model = Decoder(
        num_norm_groups=main_config.num_norm_groups,
        intermediate_features=main_config.decoder_intermediate_features,
        block_depth=main_config.block_depth,
        kernel_dim=main_config.kernel_dim,
        dtype=jnp.bfloat16
    )
    autoencoder_model = Autoencoder(encoder=encoder_model, decoder=decoder_model)

    x = preprocess(
        x=jnp.ones((main_config.batch_size, main_config.num_field_params), dtype=jnp.float32),
        train_on_hash_grid=main_config.train_on_hash_grid,
        hash_grid_end=main_config.num_hash_grid_params,
        left_padding=main_config.left_padding,
        right_padding=main_config.right_padding,
        requires_padding=main_config.requires_padding
    )
    params_key = jax.random.PRNGKey(main_config.model_seed)
    params = autoencoder_model.init(params_key, x=x, train=False)['params']
    opt = optax.adamw(
        learning_rate=main_config.learning_rate, weight_decay=main_config.weight_decay
    )
    state = TrainState.create(apply_fn=autoencoder_model.apply, params=params, tx=opt)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    main_config_dict['param_count'] = param_count
    print(f'param_count {param_count:,}')
    
    num_train_samples = len(train_set)
    train_steps = num_train_samples // main_config.batch_size
    print('num_train_samples', num_train_samples)
    print('train_steps', train_steps)
    num_test_samples = len(test_set)
    test_steps = num_test_samples // main_config.batch_size
    print('num_test_samples', num_test_samples)
    print('test_steps', test_steps)
    
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=True))
    if checkpoint_path is not None:
        print(f'loading checkpoint {checkpoint_path}')
        checkpointer.restore(os.path.abspath(checkpoint_path), item=state)

    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    wandb.init(project='conv-vae', config=main_config_dict)
    num_preview_samples = 5
    print('num_preview_samples', num_preview_samples)
    preview_samples = test_set[:num_preview_samples]['params']
    print('preview_samples shape', preview_samples.shape)
    for epoch in range(main_config.num_epochs):
        train_set = train_set.shuffle(seed=epoch)
        train_iterator = train_set.iter(main_config.batch_size)
        losses_this_epoch = []
        for step in range(train_steps):
            batch = next(train_iterator)['params']
            loss, state = train_step(
                state=state, 
                batch=batch, 
                train_on_hash_grid=main_config.train_on_hash_grid, 
                hash_grid_end=main_config.num_hash_grid_params, 
                left_padding=main_config.left_padding,
                right_padding=main_config.right_padding, 
                requires_padding=main_config.requires_padding
            )
            losses_this_epoch.append(loss)
        average_loss = sum(losses_this_epoch) / len(losses_this_epoch)
        print(f'epoch {epoch}, loss {average_loss}')
        wandb.log({'loss': average_loss}, step=state.step)
        
        test_set = test_set.shuffle(seed=epoch)
        test_iterator = test_set.iter(main_config.batch_size)
        losses_this_test = []
        for step in range(test_steps):
            batch = next(test_iterator)['params']
            loss = test_step(
                state=state, 
                batch=batch, 
                train_on_hash_grid=main_config.train_on_hash_grid, 
                hash_grid_end=main_config.num_hash_grid_params, 
                left_padding=main_config.left_padding, 
                right_padding=main_config.right_padding, 
                requires_padding=main_config.requires_padding
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
            state=state, 
            batch=preview_samples, 
            train_on_hash_grid=main_config.train_on_hash_grid, 
            hash_grid_end=main_config.num_hash_grid_params, 
            left_padding=main_config.left_padding, 
            right_padding=main_config.right_padding, 
            requires_padding=main_config.requires_padding
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
