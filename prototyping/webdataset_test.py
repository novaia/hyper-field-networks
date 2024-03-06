from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from jax import numpy as jnp
import os
import matplotlib.pyplot as plt

def get_data_iterator(batch_size):
    @pipeline_def(batch_size=batch_size, num_threads=4, device_id=0)
    def wds_pipeline():
        raw_image, ascii_label = fn.readers.webdataset(
            paths='data/devel-0.tar', 
            ext=["jpg", "cls"], 
            missing_component_behavior="error"
        )
        image = fn.decoders.image(raw_image)
        ascii_shift = types.Constant(48).uint8()
        label = fn.cast(ascii_label, dtype=types.UINT8) - ascii_shift
        return image, label

    data_pipeline = wds_pipeline()
    data_iterator = DALIGenericIterator(
        pipelines=[data_pipeline], 
        output_map=['x', 'y'], 
        last_batch_policy=LastBatchPolicy.DROP
    )
    return data_iterator

def main():
    data_iterator = get_data_iterator(32)
    xy = next(data_iterator)
    x = xy['x']
    y = xy['y']
    print(x.shape)
    print(y.shape)
    print(y)
    out_path = 'data/dali_loader_test'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(32):
        image = x[i]
        label = y[i]
        plt.imsave(
            os.path.join(out_path, f'label{label}_image{i}.png'), 
            image, 
            cmap='gray'
        )

if __name__ == '__main__':
    main()
