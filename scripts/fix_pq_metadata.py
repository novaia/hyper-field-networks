import pyarrow as pa
import pyarrow.parquet as pq
import json

table = pq.read_table('../0201.parquet')

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
pq.write_table(new_table, '../0201.parquet')
