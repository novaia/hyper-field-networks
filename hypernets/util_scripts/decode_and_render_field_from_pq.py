# Decodes and renders split field latents from merged latents dataset.
import os, json, argparse
from safetensors.flax import load_file

import numpy as np
from flax import traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fields.common.flattening import unflatten_params
from fields import ngp_image
from hypernets.common.pytree_utils import move_pytree_to_gpu
from hypernets import split_field_conv_ae
from hypernets.split_field_conv_ae import (
    SplitFieldConvAeConfig, init_model_from_config, preprocess
)

def load_dataset(dataset_path, test_size, split_seed, max_pq_files):
    field_config = None
    with open(os.path.join(dataset_path, 'field_config.json'), 'r') as f:
        field_config = json.load(f)
    assert field_config is not None
    
    param_map = None
    with open(os.path.join(dataset_path, 'param_map.json'), 'r') as f:
        param_map = json.load(f)
    assert param_map is not None
    
    hash_ae_config_dict = None
    with open(os.path.join(dataset_path, 'hash_ae_config.json'), 'r') as f:
        hash_ae_config_dict = json.load(f)
    assert hash_ae_config_dict is not None
    hash_ae_config = split_field_conv_ae.SplitFieldConvAeConfig(hash_ae_config_dict)

    mlp_ae_config_dict = None
    with open(os.path.join(dataset_path, 'mlp_ae_config.json'), 'r') as f:
        mlp_ae_config_dict = json.load
    assert mlp_ae_config_dict is not None
    mlp_ae_config = split_field_conv_ae.SplitFieldConvAeConfig(mlp_ae_config_dict)
    
    parquet_dir = os.path.join(dataset_path, 'data')
    parquet_paths = [
        os.path.join(parquet_dir, p) 
        for p in os.listdir(parquet_dir) if p.endswith('.parquet')
    ]
    num_parquet_files = len(parquet_paths)
    assert num_parquet_files > 0
    print(f'Found {num_parquet_files} parquet file(s) in dataset directory')
    if num_parquet_files > max_pq_files:
        parquet_paths = parquet_paths[:max_pq_files]
        print(f'Only using {max_pq_files} out of {num_parquet_files} parquet file(s)')

    dataset = datasets.load_dataset('parquet', data_files=parquet_paths)
    train, test = dataset['train'].train_test_split(test_size=test_size, seed=split_seed).values()
    device = str(jax.devices('gpu')[0])
    train = train.with_format('jax', device=device)
    test = test.with_format('jax', device=device)
    return train, test, field_config, param_map, hash_ae_config, mlp_ae_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash_decoder', type=str, required=True)
    parser.add_argument('--mlp_decoder', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--max_pq_files', type=int, default=1)
    parser.add_argument('--output', type=int, default=1)
    args = parser.parse_args()

    train_set, test_set, field_config, field_param_map, hash_ae_config, mlp_ae_config = \
        load_dataset(dataset_path=args.dataset, split_size=0.1, split_seed=0, max_pq_files=args.max_pq_files)

    _, _, hash_decoder_model = init_model_from_config(hash_ae_config)
    hash_decoder_params_cpu = traverse_util.unflatten_dict(load_file(args.hash_decoder), sep='.')
    hash_decoder_params = move_pytree_to_gpu(hash_decoder_params_cpu)

    _, _, mlp_decoder_model = init_model_from_config(mlp_ae_config)
    mlp_decoder_params_cpu = traverse_util.unflatten_dict(load_file(args.mlp_decoder), sep='.')
    mlp_decoder_params = move_pytree_to_gpu(mlp_decoder_params_cpu)
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    data_iterator = train_set.iter(batch_size=1)
    for i in range(args.n_samples):
        batch = next(data_iterator)
        mlp_latents = split_field_conv_ae.preprocess( 
            x=batch['mlp_latents'],
            train_on_hash_grid=mlp_ae_config.train_on_hash_grid,
            left_padding=mlp_ae_config.left_padding,
            right_padding=mlp_ae_config.right_padding,
            requires_padding=mlp_ae_config.requires_padding
        )
        hash_latents = split_field_conv_ae.preprocess( 
            x=batch['hash_latents'],
            train_on_hash_grid=hash_ae_config.train_on_hash_grid,
            left_padding=hash_ae_config.left_padding,
            right_padding=hash_ae_config.right_padding,
            requires_padding=hash_ae_config.requires_padding
        )
        hash_params = hash_decoder_model.apply({'params': hash_decoder_params}, x=hash_latents)[0]
        mlp_params = mlp_decoder_model.apply({'params': mlp_decoder_params}, x=mlp_latents)[0]
        field_params = jnp.concatenate([hash_params, mlp_params], axis=0)
        field_params = unflatten_params(field_params, field_param_map)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], 
            field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(args.output, f'render_{i}.png'), field_render)

if __name__ == '__main__':
    main()
