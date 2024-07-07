import glob
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq
import json, os, argparse

def get_dataset(dataset_path):
    if dataset_path.endswith('/'):
        glob_pattern = f'{dataset_path}*.parquet'
    else:
        glob_pattern = f'{dataset_path}/*.parquet'
    parquet_files = glob.glob(glob_pattern)
    assert len(parquet_files) > 0, 'No parquet files were found in dataset directory.'
    print(f'Found {len(parquet_files)} parquet files in dataset directory.')
    dataset = load_dataset(
        'parquet', 
        data_files={'train': parquet_files},
        split='train',
        num_proc=1
    )
    return dataset

def save_merged_table(
    merged_table_data, num_mlp_latents, num_hash_latents, table_number, output_path, zfill_amount
):
    print(f'Entries in table {table_number}: {len(merged_table_data)}')
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
    table = pa.Table.from_pylist(merged_table_data, schema=schema)
    pq.write_table(table, f'{output_path}/{str(table_number).zfill(zfill_amount)}.parquet')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mlp', type=str, required=True, help='Path to dataset containing MLP latents')
    parser.add_argument('--hash', type=str, required=True, help='Path to dataset containing hash grid latents')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory for merged dataset')
    args = parser.parse_args()

    mlp_dataset = get_dataset(args.mlp)
    hash_dataset = get_dataset(args.hash)

    assert len(mlp_dataset) == len(hash_dataset), 'Datasets have different lengths'

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    num_samples = len(mlp_dataset)
    mlp_iterator = mlp_dataset.iter(batch_size=1)
    hash_iterator = hash_dataset.iter(batch_size=1)
    print('Samples in dataset:', num_samples)

    samples_per_table = 30000
    pq_table_data = []
    current_table_index = 0
    
    for _ in range(num_samples):
        mlp_latent = next(mlp_iterator)['latents'][0]
        hash_latent = next(hash_iterator)['latents'][0]
        pq_table_data.append({
            'mlp_latents': mlp_latent,
            'hash_latents': hash_latent
        })
        if len(pq_table_data) >= samples_per_table:
            num_mlp_latents = len(pq_table_data[0]['mlp_latents'])
            num_hash_latents = len(pq_table_data[0]['hash_latents'])
            save_merged_table(
                merged_table_data=pq_table_data,
                num_mlp_latents=num_mlp_latents,
                num_hash_latents=num_hash_latents,
                table_number=current_table_index,
                output_path=args.output,
                zfill_amount=4
            )
            pq_table_data = []
            current_table_index += 1
    
    if len(pq_table_data) > 0:
        num_mlp_latents = len(pq_table_data[0]['mlp_latents'])
        num_hash_latents = len(pq_table_data[0]['hash_latents'])
        save_merged_table(
            merged_table_data=pq_table_data, 
            num_mlp_latents=num_mlp_latents, 
            num_hash_latents=num_hash_latents, 
            table_number=current_table_index,
            output_path=args.output, 
            zfill_amount=4
        )

    print(f"Combined dataset saved to {output_path}")

if __name__ == '__main__':
    main()
