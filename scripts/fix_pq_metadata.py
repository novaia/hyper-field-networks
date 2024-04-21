import pyarrow as pa
import pyarrow.parquet as pq
import json, argparse, os

def fix_metadata_and_save(pq_file_name, input_path, output_path):
    table = pq.read_table(os.path.join(input_path, pq_file_name))
    metadata = {
        b'huggingface': json.dumps({
            'info': {
                'features': {
                    'params': {'_type': 'Sequence', 'feature': {'_type': 'Value', 'dtype': 'float32'}},
                    'image': {'_type': 'Image'}
                }
            }
        }).encode('utf-8')
    }

    new_schema = table.schema.with_metadata(metadata)
    new_table = pa.Table.from_arrays(table.columns, schema=new_schema)
    pq.write_table(new_table, os.path.join(output_path, pq_file_name))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    
    assert os.path.exists(args.input_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pq_file_list = [p for p in os.listdir(args.input_path) if p.endswith('.parquet')]
    print(pq_file_list)
    for file_name in pq_file_list:
        fix_metadata_and_save(file_name, args.input_path, args.output_path)
        print(f'Processed {file_name}')

if __name__ == '__main__':
    main()
