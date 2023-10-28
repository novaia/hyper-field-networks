import os
import argparse
from natsort import natsorted
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--chunk_size', type=int, required=True)
args = parser.parse_args()

dir_list = natsorted(os.listdir(args.dir))
num_chunks = int(math.ceil(len(dir_list) / args.chunk_size))
for chunk_id in range(num_chunks):
    chunk_start = chunk_id * args.chunk_size
    chunk_end = chunk_start + args.chunk_size-1
    chunk_name = f'{chunk_start}-{chunk_end}'
    print(f'Chunking {chunk_name}')
    os.makedirs(os.path.join(args.dir, chunk_name))
    for i in range(chunk_start, min(chunk_end+1, len(dir_list))):
        source_path = os.path.join(args.dir, dir_list[i])
        target_path = os.path.join(args.dir, chunk_name, dir_list[i])
        os.rename(source_path, target_path)