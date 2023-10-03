# Import the modules
import gzip
import msgpack

# Define the file name
file_name = "data/base.ingp"

# Open the file in binary mode
with gzip.open(file_name, "rb") as f:
    # Read the file content
    file_content = f.read()
    file_content = msgpack.unpackb(file_content)
    print(file_content.keys())
    print(file_content['network'].keys())
    print(file_content['snapshot'].keys())
    print(file_content['snapshot']['params_type'])
    print(file_content['snapshot']['n_params'])
    params = msgpack.unpackb(file_content['snapshot']['params_binary'])
    print(params)
    
    #file_content = gzip.decompress(file_content)
    
    # Unpack the file content as a msgpack object
    # Print the file content
    #print(file_content)
