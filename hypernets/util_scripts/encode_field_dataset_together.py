import os
import json
import argparse
import glob
from safetensors.flax import load_file
from hypernets.split_field_conv_ae import (
    SplitFieldConvAeConfig, init_model_from_config, preprocess
)
import numpy as np
from flax import traverse_util
import jax
import jax.numpy as jnp
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
from hypernets.common.pytree_utils import move_pytree_to_gpu

def get_dataset(dataset_path):
    if dataset_path.endswith('/'):
        glob_pattern = f'{dataset_path}data/*.parquet'
    else:
        glob_pattern = f'{dataset_path}/data/*.parquet'
    parquet_files = glob.glob(glob_pattern)
    assert len(parquet_files) > 0, 'No parquet files were found in dataset directory.'
    print(f'Found {len(parquet_files)} parquet files in dataset directory.')
    dataset = load_dataset(
        'parquet', 
        data_files={'train': parquet_files},
        split='train',
        num_proc=8
    )
    dataset = dataset.with_format('jax')
    return dataset

def save_latents_table(table_data, num_mlp_latents, num_hash_latents, table_number, output_path, zfill_amount):
    print(f'Entries in table {table_number}: {len(table_data)}')
    schema = pa.schema(
        fields=[
            ('mlp_latents', pa.list_(pa.float32(), list_size=num_mlp_latents)),
            ('hash_latents', pa.list_(pa.float32(), list_size=num_hash_latents)),
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'mlp_latents': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                        'hash_latents': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(table_data, schema=schema)
    pq.write_table(table, f'{output_path}/{str(table_number).zfill(zfill_amount)}.parquet')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp_config', type=str, required=True)
    parser.add_argument('--hash_config', type=str, required=True)
    parser.add_argument('--mlp_encoder', type=str, required=True)
    parser.add_argument('--hash_encoder', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--from_npy', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    with open(args.mlp_config, 'r') as f:
        mlp_model_config = SplitFieldConvAeConfig(json.load(f))
    _, mlp_encoder_model, _ = init_model_from_config(mlp_model_config)
    if not args.from_npy:
        mlp_encoder_params_cpu = traverse_util.unflatten_dict(load_file(args.mlp_encoder), sep='.')
    else:
        mlp_encoder_params_cpu = np.load(args.mlp_encoder, allow_pickle=True).tolist()
    mlp_encoder_params = move_pytree_to_gpu(mlp_encoder_params_cpu)

    with open(args.hash_config, 'r') as f:
        hash_model_config = SplitFieldConvAeConfig(json.load(f))
    _, hash_encoder_model, _ = init_model_from_config(hash_model_config)
    if not args.from_npy:
        hash_encoder_params_cpu = traverse_util.unflatten_dict(load_file(args.hash_encoder), sep='.')
    else:
        hash_encoder_params_cpu = np.load(args.hash_encoder, allow_pickle=True).tolist()
    hash_encoder_params = move_pytree_to_gpu(hash_encoder_params_cpu)
    
    dataset = get_dataset(args.dataset)
    batch_size = 1
    dataset_iterator = dataset.iter(batch_size=batch_size)
    num_samples = len(dataset)
    num_batches = num_samples // batch_size 
    print('Samples in dataset:', num_samples)
    print('Batches in dataset:', num_batches)

    #samples_per_table = 30000
    samples_per_table = 30
    current_table_index = 0
    pq_table_data = []

    for batch_id in range(num_batches):
        batch_samples = next(dataset_iterator)['params']
        batch_samples = jax.device_put(jnp.array(batch_samples), device=jax.devices('gpu')[0])
        
        mlp_batch_samples = preprocess(
            x=batch_samples,
            train_on_hash_grid=mlp_model_config.train_on_hash_grid,
            hash_grid_end=mlp_model_config.num_hash_grid_params,
            left_padding=mlp_model_config.left_padding,
            right_padding=mlp_model_config.right_padding,
            requires_padding=mlp_model_config.requires_padding
        )
        mlp_batch_latents_gpu = mlp_encoder_model.apply({'params': mlp_encoder_params}, x=mlp_batch_samples)
        mlp_batch_latents = np.array(
            jax.device_put(mlp_batch_latents_gpu, device=jax.devices('cpu')[0]), 
            dtype=np.float32
        )
        
        hash_batch_samples = preprocess(
            x=batch_samples,
            train_on_hash_grid=hash_model_config.train_on_hash_grid,
            hash_grid_end=hash_model_config.num_hash_grid_params,
            left_padding=hash_model_config.left_padding,
            right_padding=hash_model_config.right_padding,
            requires_padding=hash_model_config.requires_padding
        )
        hash_batch_latents_gpu = hash_encoder_model.apply({'params': hash_encoder_params}, x=hash_batch_samples)
        hash_batch_latents = np.array(
            jax.device_put(hash_batch_latents_gpu, device=jax.devices('cpu')[0]), 
            dtype=np.float32
        )

        for latent_id in range(batch_size):
            mlp_latent = np.ravel(mlp_batch_latents[latent_id])
            hash_latent = np.ravel(hash_batch_latents[latent_id])

            pq_row_data = {
                'mlp_latents': mlp_latent.tolist(),
                'hash_latents': hash_latent.tolist(),
            }
            pq_table_data.append(pq_row_data)

            if len(pq_table_data) >= samples_per_table:
                save_latents_table(
                    table_data=pq_table_data, 
                    num_mlp_latents=mlp_latent.shape[0], 
                    num_hash_latents=hash_latent.shape[0],
                    table_number=current_table_index, 
                    output_path=args.output, 
                    zfill_amount=4
                )
                pq_table_data = []
                current_table_index += 1
                exit()

    if len(pq_table_data) > 0:
        save_latents_table(
            pq_table_data, 
            len(pq_table_data[0]['mlp_latents']), 
            current_table_index, 
            args.output_path, 
            4
        )

    print('Finished processing all samples')

if __name__ == '__main__':
    main()
