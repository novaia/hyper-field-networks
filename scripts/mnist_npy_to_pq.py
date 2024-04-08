import os, json, io
import pyarrow as pa
import pyarrow.parquet as pq
import jax.numpy as jnp
import numpy as np
from PIL import Image
from fields.common.flattening import generate_param_map, flatten_params

def main():
    input_dir = 'data/mnist_ingp'
    output_dir = 'data/mnist_ingp_parquet'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_paths = os.listdir(input_dir)
    input_paths_npy = [p for p in input_paths if p.endswith('.npy')]
    assert len(input_paths_npy) > 0

    params = dict(jnp.load(os.path.join(input_dir, input_paths_npy[0]), allow_pickle=True).tolist())
    param_map, param_count = generate_param_map(params)
    with open(os.path.join(output_dir, 'param_map.json'), 'w') as f:
        json.dump(param_map, f, indent=4)
    
    pq_table_data = []
    for i, partial_npy_path in enumerate(input_paths_npy):
        npy_path = os.path.join(input_dir, partial_npy_path)
        params = dict(jnp.load(npy_path, allow_pickle=True).tolist())
        flat_params = flatten_params(params, param_map, param_count)
        png_path = npy_path[:-4] + '.png'
        image = Image.open(png_path)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        pq_row_data = {
            'params': np.array(flat_params, dtype=np.float32).tolist(),
            'image': {
                'bytes': image_bytes.getvalue(),
                'path': f'{i}.png'
            }
        }
        pq_table_data.append(pq_row_data)
    
    schema = pa.schema(
        fields=[
            ('params', pa.list_(pa.float32(), list_size=param_count)),
            ('image', pa.struct([('bytes', pa.binary()), ('path', pa.string())]))
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'params': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                        'image': {'_type': 'Image'}
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(pq_table_data, schema=schema)
    pq.write_table(table, f'{output_dir}/{0000}.parquet')

if __name__ == '__main__':
    main()
