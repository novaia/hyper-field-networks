# Decodes and renders split field latents from merged latents dataset.
import os, json, argparse
from safetensors.flax import load_file
import datasets
import numpy as np
from flax import traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fields.common.flattening import unflatten_params
from fields import ngp_image
from hypernets.common.pytree_utils import move_pytree_to_gpu
from hypernets import split_field_conv_ae

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
        mlp_ae_config_dict = json.load(f)
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
    train = dataset['train']    
    test = None
    device = str(jax.devices('gpu')[0])
    train = train.with_format('jax', device=device)
    #test = test.with_format('jax', device=device)
    return train, test, field_config, param_map, hash_ae_config, mlp_ae_config

def preprocess(latent, latent_channels):
    return jnp.reshape(latent, [latent.shape[0], latent.shape[1]//latent_channels, latent_channels])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hash_decoder', type=str, required=True)
    parser.add_argument('--mlp_decoder', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--max_pq_files', type=int, default=1)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--from_npy', action='store_true')
    # TODO: include this information in dataset.
    parser.add_argument('--latent_channels', type=int, default=4)
    args = parser.parse_args()

    train_set, test_set, field_config, field_param_map, hash_ae_config, mlp_ae_config = \
        load_dataset(dataset_path=args.dataset, test_size=0.0, split_seed=0, max_pq_files=args.max_pq_files)

    _, _, hash_decoder_model = split_field_conv_ae.init_model_from_config(hash_ae_config)
    if not args.from_npy:
        hash_decoder_params_cpu = traverse_util.unflatten_dict(load_file(args.hash_decoder), sep='.')
    else:
        hash_decoder_params_cpu = np.load(args.hash_decoder, allow_pickle=True).tolist()
    hash_decoder_params = move_pytree_to_gpu(hash_decoder_params_cpu)

    _, _, mlp_decoder_model = split_field_conv_ae.init_model_from_config(mlp_ae_config)
    if not args.from_npy:
        mlp_decoder_params_cpu = traverse_util.unflatten_dict(load_file(args.mlp_decoder), sep='.')
    else:
        mlp_decoder_params_cpu = np.load(args.mlp_decoder, allow_pickle=True).tolist()
    mlp_decoder_params = move_pytree_to_gpu(mlp_decoder_params_cpu)
    
    field_model = ngp_image.create_model_from_config(field_config)
    field_state = ngp_image.create_train_state(field_model, 3e-4, jax.random.PRNGKey(0))

    data_iterator = train_set.iter(batch_size=1)
    for i in range(args.n_samples):
        batch = next(data_iterator)
        hash_latents = preprocess(latent=batch['hash_latents'], latent_channels=args.latent_channels)
        hash_latents = jnp.array(hash_latents, dtype=jnp.bfloat16)
        hash_params = hash_decoder_model.apply({'params': hash_decoder_params}, x=hash_latents)
        hash_params = jnp.squeeze(hash_params, axis=-1)
        #hash_params = batch['hash_latents'] 
        hash_params = split_field_conv_ae.remove_padding(
            hash_params, 
            hash_ae_config.left_padding, 
            hash_ae_config.right_padding, 
            hash_ae_config.requires_padding
        )

        mlp_latents = preprocess(latent=batch['mlp_latents'], latent_channels=args.latent_channels)
        mlp_latents = jnp.array(mlp_latents, dtype=jnp.bfloat16)
        mlp_params = mlp_decoder_model.apply({'params': mlp_decoder_params}, x=mlp_latents)
        mlp_params = jnp.squeeze(mlp_params, axis=-1)
        #mlp_params = batch['mlp_latents'] 
        mlp_params = split_field_conv_ae.remove_padding(
            mlp_params, 
            mlp_ae_config.left_padding, 
            mlp_ae_config.right_padding, 
            mlp_ae_config.requires_padding
        )
        field_params = jnp.concatenate([hash_params, mlp_params], axis=-1)[0]
        field_params = jnp.array(field_params, dtype=jnp.float32)
        field_params = unflatten_params(field_params, field_param_map)
        field_state = field_state.replace(params=field_params)
        field_render = ngp_image.render_image(
            field_state, field_config['image_height'], 
            field_config['image_width'], field_config['channels']
        )
        field_render = jax.device_put(field_render, jax.devices('cpu')[0])
        plt.imsave(os.path.join(args.output, f'render_{i}.png'), field_render)

if __name__ == '__main__':
    main()
