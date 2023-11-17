from nvidia.dali import pipeline_def, fn
from nvidia.dali.plugin.jax import DALIGenericIterator
import jax
import time
import random
import jax.numpy as jnp
import os
import math

def get_dali_iterator(dataset_path, batch_size, num_threads=3):
    @pipeline_def
    def my_pipeline_def():
        data = fn.readers.numpy(
            device='cpu', 
            file_root=dataset_path, 
            file_filter='*.npy', 
            random_shuffle=True,
            name='weight_reader'
        )
        return data

    my_pipeline = my_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0)
    iterator = DALIGenericIterator(
        pipelines=[my_pipeline], output_map=['x'], reader_name='weight_reader'
    )
    return iterator

def load_batch(sample_paths, batch_size, dtype):
    random.shuffle(sample_paths)
    batch_paths = sample_paths[:batch_size]
    batch = []
    for path in batch_paths:
        batch.append(jnp.load(path))
    batch = jnp.array(batch, dtype=dtype)
    return batch

def main():
    dataset_path = 'data/ngp_images/packed_ngp_cifar10'
    batch_size = 32

    iterator = get_dali_iterator(dataset_path, batch_size)
    gpu = jax.devices('gpu')[0]
    start_time = time.time()
    for batch in iterator:
        batch = jax.device_put(batch['x'], device=gpu)
        batch = batch * 2
    end_time = time.time()
    print('Dali load time: ', end_time - start_time)

    sample_paths = os.listdir(dataset_path)
    valid_sample_paths = []
    for path in sample_paths:
        if path.endswith('.npy'):
            valid_sample_paths.append(os.path.join(dataset_path, path))
    num_batches = math.ceil(len(valid_sample_paths)/batch_size)
    start_time = time.time()
    for i in range(num_batches):
        batch = load_batch(valid_sample_paths, batch_size, jnp.float32)  
        batch = batch * 2
    end_time = time.time()
    print('Naive load time: ', end_time - start_time)

if __name__ == '__main__':
    main()