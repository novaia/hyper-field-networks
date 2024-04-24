from datasets import load_dataset
from float_tokenization import tokenize
import os, argparse, json
import pyarrow as pa
import pyarrow.parquet as pq
import jax
import jax.numpy as jnp
import numpy as np

def save_table(table_data, output_path):
    schema = pa.schema(
        fields=[
            ('tokens', pa.list_(pa.uint32(), list_size=len(table_data[0]['tokens']))),
            ('image', pa.struct([('bytes', pa.binary()), ('path', pa.string())]))
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'tokens': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'uint32'}},
                        'image': {'_type': 'Image'}
                    }
                }
            }).encode('utf-8')
        }
    )
    table = pa.Table.from_pylist(table_data, schema=schema)
    pq.write_table(table, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pq_paths = [p for p in os.listdir(args.input_path) if p.endswith('.parquet')]
    assert len(pq_paths) > 0, 'No parquet files were found in input directory.'    
    
    for current_path in pq_paths:
        table = pq.read_table(os.path.join(args.input_path, current_path)) 
        jitted_tokenize = jax.jit(tokenize)
        pq_table_data = []
        for i in range(len(table)):
            params = jnp.array(pa.array(table['params'][i]).to_numpy(), dtype=jnp.float16)
            tokens = jnp.array(jitted_tokenize(params), dtype=jnp.uint32)
            image = table['image'][i]
            pq_row_data = {
                'tokens': tokens.tolist(),
                'image': {
                    'bytes': image['bytes'],
                    'path': image['path']
                }
            }
            pq_table_data.append(pq_row_data)
        save_table(pq_table_data, os.path.join(args.output_path, current_path))
        print(f'Processed {current_path}')

if __name__ == '__main__':
    main()
