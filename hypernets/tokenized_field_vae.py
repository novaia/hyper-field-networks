import jax
from jax import numpy as jnp
import datasets
import os, json

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

    dataset = datasets.load_dataset(
        'parquet', 
        data_files={'train': parquet_paths},
        split='train',
        num_proc=1
    )
    dataset = dataset.with_format('jax')
    return dataset, field_config, param_map

def main():
    dataset_path = 'data/mnist-ngp-image-612-11bit'
    dataset, field_config, param_map = load_dataset(dataset_path)
    print(len(dataset))
    print(field_config)
    print(param_map)

if __name__ == '__main__':
    main()
