import os
import sys
sys.path.append(os.getcwd())
import jax.numpy as jnp
import argparse
import json
from hypernets.packing.ngp_nerf import pack_weights, unpack_weights

def get_file_paths_in_subdirs(dir):
    if os.path.isdir(dir):
        files = []
        for subdir in os.listdir(dir):
            subdir_path = os.path.join(dir, subdir)
            if os.path.isdir(subdir_path):
                files.extend(get_file_paths_in_subdirs(subdir_path))
            else:
                files.append(subdir_path)
        return files
    raise ValueError('The specified directory does not exist')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--unpack', type=bool, default=False)
    parser.add_argument(
        '--rm', type=bool, default=False, help='Remove original files after packing'
    )
    parser.add_argument('--loss_threshold', type=float, default=1e-3)
    args = parser.parse_args()
    assert args.input_dir is not None, 'Must specify a directory'
    assert args.unpack == False, 'Unpacking is not yet implemented'

    file_paths = get_file_paths_in_subdirs(args.input_dir)
    for path in file_paths:
        params = jnp.load(path, allow_pickle=True).tolist()
        print(f'Loaded {path}')
        if params['final_loss'] > args.loss_threshold:
            print('Loss is too high, skipping')
            continue
        params = params['params']
        packed_params, param_map = pack_weights(params, 64)
        save_path = os.path.join(args.output_dir, os.path.basename(path))
        jnp.save(save_path, packed_params)
        print(f'Saved to {save_path}')
        if args.rm:
            os.remove(path)
            print(f'Removed {path}')
    with open(os.path.join(args.output_dir, 'param_map.json'), 'w+') as f:
        json.dump(param_map, f, indent=4)

if __name__ == '__main__':
    main()