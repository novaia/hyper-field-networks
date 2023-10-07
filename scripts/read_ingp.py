import gzip
import msgpack

file_name = "data/base.ingp"

with gzip.open(file_name, "rb") as f:
    file_content = f.read()
    file_content = msgpack.unpackb(file_content)
    print(file_content.keys())
    print(file_content['network'].keys())
    print(file_content['snapshot'].keys())
    print(file_content['snapshot']['params_type'])
    print(file_content['snapshot']['n_params'])
    params = msgpack.unpackb(file_content['snapshot']['params_binary'])
    print(params)
