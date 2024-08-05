import os, json, argparse
from safetensors.flax import load_file
from hypernets.split_field_conv_ae import (
    SplitFieldConvAeConfig, init_model_from_config, preprocess, remove_padding
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
    parser.add_argument('--decoder', type=str, required=True)
    parser.add_argument('--sample', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        model_config = SplitFieldConvAeConfig(json.load(f))
    _, encoder_model, decoder_model = init_model_from_config(model_config)
    
    sample_cpu = np.load(args.sample, allow_pickle=False)
    sample = jax.device_put(jnp.array(sample_cpu), device=jax.devices('gpu')[0])
    encoder_params_cpu = traverse_util.unflatten_dict(load_file(args.encoder), sep='.')
    encoder_params = move_pytree_to_gpu(encoder_params_cpu)
    decoder_params_cpu = traverse_util.unflatten_dict(load_file(args.decoder), sep='.')
    decoder_params = move_pytree_to_gpu(decoder_params_cpu)
    
    sample_prev = jnp.array(sample)
    sample = jnp.expand_dims(sample, axis=0)
    if model_config.train_on_hash_grid:
        other_part = sample[..., model_config.num_hash_grid_params:]
    else:
        other_part = sample[..., :model_config.num_hash_grid_params]
    sample = preprocess(
        x=sample,
        train_on_hash_grid=model_config.train_on_hash_grid,
        hash_grid_end=model_config.num_hash_grid_params,
        left_padding=model_config.left_padding,
        right_padding=model_config.right_padding,
        requires_padding=model_config.requires_padding
    )
    latent = encoder_model.apply({'params': encoder_params}, x=sample)
    re_sample = decoder_model.apply({'params': decoder_params}, x=latent)
    re_sample = jnp.squeeze(re_sample, axis=-1)
    re_sample = remove_padding(
        x=re_sample, 
        left_padding=model_config.left_padding, 
        right_padding=model_config.right_padding, 
        requires_padding=model_config.requires_padding
    )
    if model_config.train_on_hash_grid:
        re_full = jnp.concatenate([re_sample, other_part], axis=-1)
    else:
        re_full = jnp.concatenate([other_part, re_sample], axis=-1)
    mse = jnp.mean((re_full - sample_prev)**2)
    print('mse:', mse)
    #indices = np.arange(latent.shape[0])
    #plt.figure(figsize=(10, 6))
    #plt.scatter(indices, latent, s=10)
    #plt.xlabel('Index')
    #plt.ylabel('Latent Value')
    #plt.title('Scatter Plot of Latent Values')
    #plt.grid(True)
    #plt.show()

if __name__ == '__main__':
    main()
