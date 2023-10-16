import os
import sys
sys.path.append(os.getcwd())
from fields import image_field, ngp_nerf, tiny_nerf, vanilla_nerf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--downscale_factor', type=int, default=1)
    args = parser.parse_args()
    if args.field == 'image_field':
        image_field.main()
    elif args.field == 'ngp_nerf':
        ngp_nerf.main(args.dataset_path, args.downscale_factor)
    elif args.field == 'tiny_nerf':
        tiny_nerf.main()
    elif args.field == 'vanilla_nerf':
        vanilla_nerf.main()
    else:
        raise ValueError(f'Unknown field: {args.field}')