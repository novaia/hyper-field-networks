# Script to extract preview samples from parquet datasets so that
# they can be inspected individually.

import os, argparse
import datasets
import numpy as np

def load_sample_dataset(dataset_path):
    parquet_dir = os.path.join(dataset_path, 'data')
    parquet_paths = [
        os.path.join(parquet_dir, p) 
        for p in os.listdir(parquet_dir) if p.endswith('.parquet')
    ]
    num_parquet_files = len(parquet_paths)
    assert num_parquet_files > 0
    print(f'Found {num_parquet_files} parquet file(s) in dataset directory')

    dataset = datasets.load_dataset('parquet', data_files=parquet_paths)
    _, sample_dataset = dataset['train'].train_test_split(test_size=0.1, seed=0).values()
    sample_dataset = sample_dataset.with_format('numpy')
    return sample_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--samples', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    dataset = load_sample_dataset(args.dataset)
    for i in range(args.samples):
        np.save(
            file=os.path.join(args.output, f'{i}.npy'),
            arr=dataset[i]['params'], 
            allow_pickle=False
        )

if __name__ == '__main__':
    main()
