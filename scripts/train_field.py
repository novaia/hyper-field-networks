import os
import sys
sys.path.append(os.getcwd())
from fields import image_field, instant_nerf, tiny_nerf, vanilla_nerf
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str)
    args = parser.parse_args()
    if args.field == 'image_field':
        image_field.main()
    elif args.field == 'instant_nerf':
        instant_nerf.main()
    elif args.field == 'tiny_nerf':
        tiny_nerf.main()
    elif args.field == 'vanilla_nerf':
        vanilla_nerf.main()
    else:
        raise ValueError(f'Unknown field: {args.field}')