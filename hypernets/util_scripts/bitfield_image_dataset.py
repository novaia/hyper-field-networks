import os, argparse, json
import jax
import jax.numpy as jnp
import numpy as np
import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from fp_tokenization import fp32_to_bitfield16, bitfield16_to_fp32
import matplotlib.pyplot as plt

# python -m hypernets.util_scripts.bitfield_image_dataset --dataset data/mnist --out data/mnist-bitfield16 --samples_per_table 70000

def load_dataset(dataset_path):
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
    context_length = jnp.ravel(full_dataset[0]['image']).shape[0]
    return full_dataset, context_length

def save_table(table_data, context_length, table_number, output_path, zfill_amount):
    print(f'Entries in table {table_number}: {len(table_data)}')
    schema = pa.schema(
        fields=[
            ('bitfields', pa.list_(pa.uint32(), list_size=context_length*16)),
        ],
        metadata={
            b'huggingface': json.dumps({
                'info': {
                    'features': {
                        'bitfields': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'uint32'}},
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
    parser.add_argument('--samples_per_table', type=int, default=16384)
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)
    out_data_path = os.path.join(args.out, 'data')
    if not os.path.exists(out_data_path):
        os.makedirs(out_data_path)

    dataset, context_length = load_dataset(args.dataset)
    
    num_samples = len(dataset)
    samples_per_table = args.samples_per_table
    pq_table_data = []
    samples_in_current_table = 0
    current_table_index = 0
    for i in range(num_samples):
        bitfields = fp32_to_bitfield16(jnp.ravel(dataset[i]['image']/255.0))
        
        #for k in range(context_length):
        #    print(bitfields[k*16:k*16+16])
        #image_re = jnp.reshape(bitfield16_to_fp32(bitfields), (28, 28))
        #plt.imsave('data/mnist-re-test.png', image_re)
        
        pq_row_data = {
            'bitfields': np.array(bitfields, dtype=bitfields.dtype).tolist(),
        }
        pq_table_data.append(pq_row_data)
        samples_in_current_table += 1
        if samples_in_current_table >= samples_per_table:
            save_table(pq_table_data, context_length, current_table_index, out_data_path, 4)
            pq_table_data = []
            samples_in_current_table = 0
            current_table_index += 1
    if samples_in_current_table > 0:
        save_table(pq_table_data, context_length, current_table_index, out_data_path, 4)

if __name__ == '__main__':
    main()
