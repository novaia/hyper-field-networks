import os, argparse, json
import jax
import numpy as np
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from fp_tokenization import tokenize

# python -m hypernets.util_scripts.tokenize_field_dataset --dataset data/colored-monsters-ngp-image-18k --out data/colored-monsters-ngp-image-18k-16bit

def load_dataset(dataset_path):
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
    full_dataset = dataset['train']  # Use the entire dataset without splitting

    device = str(jax.devices('gpu')[0])
    full_dataset = full_dataset.with_format('jax', device=device)
    context_length = full_dataset[0]['params'].shape[0]
    return full_dataset, field_config, param_map, context_length

def save_table(table_data, context_length, table_number, output_path, zfill_amount):
    print(f'Entries in table {table_number}: {len(table_data)}')
    schema = pa.schema(
        fields=[
            ('tokens', pa.list_(pa.uint32(), list_size=context_length)),
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'tokens': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'uint32'}},
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(table_data, schema=schema)
    pq.write_table(table, f'{output_path}/{str(table_number).zfill(zfill_amount)}.parquet')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    out_data_path = os.path.join(args.out, 'data')
    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)

    dataset, field_config, param_map, context_length = load_dataset(args.dataset)
    
    with open(os.path.join(args.out, 'field_config.json'), 'w') as f:
        json.dump(obj=field_config, fp=f)

    with open(os.path.join(args.out, 'param_map.json'), 'w') as f:
        json.dump(obj=param_map, fp=f)
    
    num_samples = len(dataset)
    samples_per_table = 8000
    pq_table_data = []
    samples_in_current_table = 0
    current_table_index = 0
    for i in range(num_samples):
        tokens = tokenize(dataset[i]['params'])
        pq_row_data = {
            'tokens': np.array(tokens, dtype=tokens.dtype).tolist(),
        }
        pq_table_data.append(pq_row_data)
        samples_in_current_table += 1
        if samples_in_current_table > samples_per_table:
            save_table(pq_table_data, context_length, current_table_index, out_data_path, 4)
            pq_table_data = []
            samples_in_current_table = 0
            current_table_index += 1
    if samples_in_current_table > 0:
        save_table(pq_table_data, context_length, current_table_index, out_data_path, 4)

if __name__ == '__main__':
    main()
