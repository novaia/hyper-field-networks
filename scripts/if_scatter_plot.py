import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt

dataset = load_dataset('parquet', data_files={'train': 'data/output.parquet'}, split='train')
dataset = dataset.with_format('jax')
params = dataset[0]['params']
indices = np.arange(len(dataset))

max_values = []
min_values = []

for i in range(len(dataset)):
    params = dataset[i]['params']
    max_values.append(np.max(params))
    min_values.append(np.min(params))

plt.figure(figsize=(8, 6))
plt.scatter(indices, max_values, color='red', label='Max Values')
plt.scatter(indices, min_values, color='blue', label='Min Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
plt.close()
