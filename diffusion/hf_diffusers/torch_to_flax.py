from safetensors.torch import load as torch_load
from safetensors.flax import save_file as flax_save_file
import jax.numpy as jnp
from .models.vae import AutoencoderKl
import json

import re

import jax
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

import argparse

def check_converted_param_keys(initial_flax_params, converted_flax_params):
    params = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, jax.local_devices(backend='cpu')[0]), converted_flax_params
    )
    params = flatten_dict(params)
    required_params = set(flatten_dict(initial_flax_params).keys())
    #shape_state = flatten_dict(initial_flax_params)
    missing_keys = required_params - set(params.keys())
    unexpected_keys = set(params.keys()) - required_params
    
    assert len(missing_keys) == 0, (
        f'The following keys are missing from the converted model: {missing_keys}'
    )
    assert len(unexpected_keys) == 0, (
        f'The following unexpected keys were found in the converted model: {unexpected_keys}'
    )

def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key

# Adapted from https://github.com/huggingface/transformers/blob/c603c80f46881ae18b2ca50770ef65fa4033eacd/src/transformers/modeling_flax_pytorch_utils.py#L69
# and https://github.com/patil-suraj/stable-diffusion-jax/blob/main/stable_diffusion_jax/convert_diffusers_to_jax.py
# Rename PyTorch param names to corresponding Flax param names and reshape tensor if necessary.
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
    # Conv norm or layer norm.
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)

    # Rename attention layers.
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

    # Embedding.
    if pt_tuple_key[-1] == "weight" and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict:
        pt_tuple_key = pt_tuple_key[:-1] + ("embedding",)
        return renamed_pt_tuple_key, pt_tensor

    # Conv layer.
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight" and pt_tensor.ndim == 4:
        pt_tensor = pt_tensor.transpose(2, 3, 1, 0)
        return renamed_pt_tuple_key, pt_tensor

    # Linear layer.
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("kernel",)
    if pt_tuple_key[-1] == "weight":
        pt_tensor = pt_tensor.T
        return renamed_pt_tuple_key, pt_tensor

    # Old PyTorch layer norm weight.
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("weight",)
    if pt_tuple_key[-1] == "gamma":
        return renamed_pt_tuple_key, pt_tensor

    # Old PyTorch layer norm bias.
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("bias",)
    if pt_tuple_key[-1] == "beta":
        return renamed_pt_tuple_key, pt_tensor

    return pt_tuple_key, pt_tensor

def convert_pytorch_param_dict_to_flax(loaded_torch_params, random_flax_params):
    # Convert PyTorch tensors to Numpy.
    loaded_torch_params = {k: v.numpy() for k, v in loaded_torch_params.items()}
    random_flax_params = flatten_dict(random_flax_params)
    converted_params = {}

    # Some parameters names need to be changed in order to match Flax names.
    for torch_key, torch_tensor in loaded_torch_params.items():
        renamed_torch_key = rename_key(torch_key)
        torch_tuple_key = tuple(renamed_torch_key.split("."))

        # Correctly rename weight parameters.
        flax_key, flax_tensor = rename_key_and_reshape_tensor(torch_tuple_key, torch_tensor, random_flax_params)

        if flax_key in random_flax_params:
            if flax_tensor.shape != random_flax_params[flax_key].shape:
                raise ValueError(
                    f'PyTorch checkpoint seems to be incorrect. Weight {torch_key} was expected to be of shape '
                    f'{random_flax_params[flax_key].shape}, but is {flax_tensor.shape}.'
                )

        # I have no idea what this comment means.
        # Also add unexpected weight so that warning is thrown.
        converted_params[flax_key] = np.asarray(flax_tensor)

    return unflatten_dict(converted_params)

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
   
    if args.model_type == 'unet':
        from .models.unet_conditional import UNet2dConditionalModel, get_model_from_config, init_model_params
    elif args.model_type == 'vae':
        from .models.autoencoder_kl import AutoencoderKl, get_model_from_config, init_model_params
    else:
        raise ValueError('Unkown model type.')

    model = get_model_from_config(config, jnp.float32)
    random_params = init_model_params(model, config, jax.random.PRNGKey(0))
    print('Finished initializing random Flax params.')
    with open(args.model_path, 'rb') as f:
        loaded_params = torch_load(f.read())
    print('Finished loading PyTorch params.')
    converted_params = convert_pytorch_param_dict_to_flax(loaded_params, random_params)
    print('Finished converting model.')
    jnp.save(args.output_path, converted_params, allow_pickle=True)
    print(f'Converted model was saved to {args.output_path}')
    check_converted_param_keys(initial_flax_params=random_params, converted_flax_params=converted_params)

if __name__ == '__main__':
    main()
