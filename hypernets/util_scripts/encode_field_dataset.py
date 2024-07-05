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

def save_latents_table(table_data, num_latents, table_number, output_path, zfill_amount):
    print(f'Entries in table {table_number}: {len(table_data)}')
    schema = pa.schema(
        fields=[
            ('latents', pa.list_(pa.float32(), list_size=num_latents)),
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'latents': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(table_data, schema=schema)
    pq.write_table(table, f'{output_path}/{str(table_number).zfill(zfill_amount)}.parquet')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--input_dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(args.config, 'r') as f:
        model_config = SplitFieldConvAeConfig(json.load(f))
    _, encoder_model, _ = init_model_from_config(model_config)
    
    params_cpu = traverse_util.unflatten_dict(load_file(args.encoder), sep='.')
    params = move_pytree_to_gpu(params_cpu)

    dataset = get_dataset(args.input_dataset)
    batch_size = 32
    dataset_iterator = dataset.iter(batch_size=batch_size)
    num_samples = len(dataset)
    num_batches = num_samples // batch_size 
    print('Samples in dataset:', num_samples)
    print('Batches in dataset:', num_batches)

    samples_per_table = 30000
    current_table_index = 0
    pq_table_data = []

    for batch_id in range(num_batches):
        batch_samples = next(dataset_iterator)['params']
        batch_samples = jax.device_put(jnp.array(batch_samples), device=jax.devices('gpu')[0])
        
        batch_samples = preprocess(
            x=batch_samples,
            train_on_hash_grid=model_config.train_on_hash_grid,
            hash_grid_end=model_config.num_hash_grid_params,
            left_padding=model_config.left_padding,
            right_padding=model_config.right_padding,
            requires_padding=model_config.requires_padding
        )
        batch_latents_gpu = encoder_model.apply({'params': params}, x=batch_samples)
        batch_latents = np.array(
            jax.device_put(batch_latents_gpu, device=jax.devices('cpu')[0]), 
            dtype=np.float32
        )
        for latent_id in range(batch_size):
            latent = np.ravel(batch_latents[latent_id])

            pq_row_data = {
                'latents': latent.tolist(),
            }
            pq_table_data.append(pq_row_data)

            if len(pq_table_data) >= samples_per_table:
                save_latents_table(pq_table_data, latent.shape[0], current_table_index, args.output_path, 4)
                pq_table_data = []
                current_table_index += 1

    if len(pq_table_data) > 0:
        save_latents_table(
            pq_table_data, 
            len(pq_table_data[0]['latents']), 
            current_table_index, 
            args.output_path, 
            4
        )

    print('Finished processing all samples')

if __name__ == '__main__':
    main()
