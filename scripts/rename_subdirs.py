import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source_dir', type=str)
parser.add_argument('--new_name', type=str)
parser.add_argument('--start_index', type=int, default=0)
args = parser.parse_args()

subdir_list = os.listdir(args.source_dir)
for i, subdir_name in enumerate(subdir_list):
    os.rename(
        os.path.join(args.source_dir, subdir_name),
        os.path.join(args.source_dir, f'{args.new_name}_{args.start_index+i}')
    )