import os
import sys
sys.path.append(os.getcwd())
from hypernets import hyper_diffusion, ngp_hyper_diffusion
from hypernets import hash_table_autoencoder
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--downscale_factor', type=int, default=1)
    args = parser.parse_args()
    if args.net == 'hyper_diffusion':
        hyper_diffusion.main()
    elif args.net == 'ngp_hyper_diffusion':
        ngp_hyper_diffusion.main()
    elif args.net == 'hash_table_autoencoder':
        hash_table_autoencoder.main()
    else:
        raise ValueError(f'Unknown network: {args.net}')