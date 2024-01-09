from safetensors.torch import load as torch_load
from safetensors.flax import save_file as flax_save_file
import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel
from .models.vae import AutoencoderKl
import json

import re

import jax
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

import argparse

def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key

# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
    """Rename PT weight names to corresponding Flax weight names and reshape tensor if necessary"""
    # conv norm or layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)

    # rename attention layers
    if len(pt_tuple_key) > 1:
        for rename_from, rename_to in (
            ("to_out_0", "proj_attn"),
            ("to_k", "key"),
            ("to_v", "value"),
            ("to_q", "query"),
        ):
            if pt_tuple_key[-2] == rename_from:
                weight_name = pt_tuple_key[-1]
                weight_name = "kernel" if weight_name == "weight" else weight_name
                renamed_pt_tuple_key = pt_tuple_key[:-2] + (rename_to, weight_name)
                if renamed_pt_tuple_key in random_flax_state_dict:
                    assert random_flax_state_dict[renamed_pt_tuple_key].shape == pt_tensor.T.shape
                    return renamed_pt_tuple_key, pt_tensor.T

    if (
        any("norm" in str_ for str_ in pt_tuple_key)
        and (pt_tuple_key[-1] == "bias")
        and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
        and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor
    elif pt_tuple_key[-1] in ["weight", "gamma"] and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict:
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
        pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        return renamed_pt_tuple_key, pt_tensor

    # conv layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # linear layer
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight":
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm weight
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # old PyTorch layer norm bias
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor

def convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model, init_key=42):
    # Step 1: Convert pytorch tensor to numpy
    pt_state_dict = {k: v.numpy() for k, v in pt_state_dict.items()}

    # Step 2: Since the model is stateless, get random Flax params
    random_flax_params = flax_model.init_weights(PRNGKey(init_key))

    random_flax_state_dict = flatten_dict(random_flax_params)
    flax_state_dict = {}

    # Need to change some parameters name to match Flax names
    for pt_key, pt_tensor in pt_state_dict.items():
        renamed_pt_key = rename_key(pt_key)
        pt_tuple_key = tuple(renamed_pt_key.split("."))

        # Correctly rename weight parameters
        flax_key, flax_tensor = rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict)

        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f"PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected to be of shape "
                    f"{random_flax_state_dict[flax_key].shape}, but is {flax_tensor.shape}."
                )

        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = jnp.asarray(flax_tensor)

    return unflatten_dict(flax_state_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['unet', 'vae'], required=True)
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))
    
    with open(args.model_path, 'rb') as f:
        model_weights = f.read()
    model_weights = torch_load(model_weights, device='cpu')

    if args.model_type == 'unet':
        model = UNet2dConditionalModel(
            sample_size=config['sample_size'],
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            down_block_types=config['down_block_types'],
            up_block_types=config['up_block_types'],
            only_cross_attention=config['dual_cross_attention'], #TODO: double check this.
            block_out_channels=config['block_out_channels'],
            layers_per_block=config['layers_per_block'],
            attention_head_dim=config['attention_head_dim'],
            cross_attention_dim=config['cross_attention_dim'],
            use_linear_projection=config['use_linear_projection'],
            flip_sin_to_cos=config['flip_sin_to_cos'],
            freq_shift=config['freq_shift']
        )
    elif args.model_type == 'vae':
        model = AutoencoderKl(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            down_block_types=config['down_block_types'],
            up_block_types=config['up_block_types'],
            layers_per_block=config['layers_per_block'],
            act_fn=config['act_fn'],
            latent_channels=config['latent_channels'],
            norm_num_groups=config['norm_num_groups'],
            sample_size=config['sample_size'],
            block_out_channels=config['block_out_channels']
        )
    else:
        raise ValueError('Unkown model type.')
    model_weights = convert_pytorch_state_dict_to_flax(model_weights, model)
    print(model_weights.keys())
    jnp.save(args.output_path, model_weights, allow_pickle=True)

if __name__ == '__main__':
    main()
