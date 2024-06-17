import os, json, argparse
from safetensors.flax import load_file
from hypernets.split_field_conv_ae import (
    SplitFieldConvAeConfig, init_model_from_config, preprocess
)
import numpy as np
from flax import traverse_util
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hypernets.common.pytree_utils import move_pytree_to_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--sample', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        model_config = SplitFieldConvAeConfig(json.load(f))
    _, encoder_model, _ = init_model_from_config(model_config)
    
    sample_cpu = np.load(args.sample, allow_pickle=False)
    sample = jax.device_put(jnp.array(sample_cpu), device=jax.devices('gpu')[0])
    params_cpu = traverse_util.unflatten_dict(load_file(args.encoder), sep='.')
    params = move_pytree_to_gpu(params_cpu)
    
    sample = jnp.expand_dims(sample, axis=0)
    sample = preprocess(
        x=sample,
        train_on_hash_grid=model_config.train_on_hash_grid,
        hash_grid_end=model_config.num_hash_grid_params,
        left_padding=model_config.left_padding,
        right_padding=model_config.right_padding,
        requires_padding=model_config.requires_padding
    )
    latent_gpu = encoder_model.apply({'params': params}, x=sample)[0]
    latent = np.array(jax.device_put(latent_gpu, device=jax.devices('cpu')[0]), dtype=np.float32)
    latent = np.ravel(latent)

    indices = np.arange(latent.shape[0])
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, latent, s=10)
    plt.xlabel('Index')
    plt.ylabel('Latent Value')
    plt.title('Scatter Plot of Latent Values')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
