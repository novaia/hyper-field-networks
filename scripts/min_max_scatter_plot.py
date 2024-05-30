import numpy as np
import datasets
import os, json
from functools import partial
import matplotlib.pyplot as plt
import math

def load_dataset(dataset_path, test_size, split_seed):
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
    return train, test

def make_plot(dataset1, title1, dataset2, title2):
    max_values1 = np.max(dataset1, axis=0)
    min_values1 = np.min(dataset1, axis=0)
    indices1 = np.arange(max_values1.shape[0])
    
    max_values2 = np.max(dataset2, axis=0)
    min_values2 = np.min(dataset2, axis=0)
    indices2 = np.arange(max_values2.shape[0])
    
    size = 0.1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.scatter(indices1, max_values1, color='red', label='Max', s=size)
    ax1.scatter(indices1, min_values1, color='blue', label='Min', s=size)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title(title1)
    ax1.legend()
    ax1.grid(True)

    ax2.scatter(indices2, max_values2, color='red', label='Max', s=size)
    ax2.scatter(indices2, min_values2, color='blue', label='Min', s=size)
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Value')
    ax2.set_title(title2)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def make_plot_old(dataset, title):
    max_values = np.max(dataset, axis=0)
    min_values = np.min(dataset, axis=0)
    indices = np.arange(max_values.shape[0])
    size = 0.1

    # Plot max and min values
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, max_values, color='red', label='Max', s=size)
    plt.scatter(indices, min_values, color='blue', label='Min', s=size)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

with open('configs/colored_monsters_ngp_image_alt.json', 'r') as f:
    field_config = json.load(f)

hash_grid_end = (
    field_config['num_hash_table_levels'] 
    * field_config['max_hash_table_entries'] 
    * field_config['hash_table_feature_dim']
)
print('hash_grid_end', hash_grid_end)

train_set, test_set = load_dataset(
    'data/colored-monsters-ngp-image-18k', 0.1, 0
)

train_set_subset = np.array(train_set[0:1500]['params'])
print(train_set_subset.shape)
hash_grid_section = train_set_subset[..., :hash_grid_end]
mlp_section = train_set_subset[..., hash_grid_end:]
make_plot(hash_grid_section, 'Hash Grid Section', mlp_section, 'MLP Section')
