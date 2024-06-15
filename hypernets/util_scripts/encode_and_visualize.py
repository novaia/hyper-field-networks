import os, json, argparse
import safetensors
from hypernets.split_field_conv_ae import (
    SplitFieldConvAeConfig, init_model_from_config
)
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--sample', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        model_config = SplitFieldConvAeConfig(json.load(f))
    _, encoder_model, _ = init_model_from_config(model_config)
    params = safetensors.flax.load_file(args.params)
    sample = np.load(args.sample, allow_pickle=False)

    # TODO:
    # - unflatten params
    # - call encoder_model.apply on sample
    # - save output to png

if __name__ == '__main__':
    main()
