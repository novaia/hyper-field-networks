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

def make_plot(dataset):
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
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

train_set, test_set = load_dataset(
    'data/colored-monsters-ngp-image-18k', 0.1, 0
)
print(train_set[0]['params'][:20])
make_plot(np.array(train_set[0:1500]['params']))
