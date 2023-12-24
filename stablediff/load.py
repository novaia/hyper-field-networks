from safetensors.torch import load
import jax.numpy as jnp
from .models.unet_conditional import UNet2dConditionalModel
import json

import re

import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.random import PRNGKey

def rename_key(key):
    regex = r"\w+[.]\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key

# Rename PyTorch weight names to corresponding Flax weight names and reshape tensor 
# if necessary.
def rename_key_and_reshape_tensor(pt_tuple_key, pt_tensor, random_flax_state_dict):
    # conv norm or layer norm
    renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
    if (
        any("norm" in str_ for str_ in pt_tuple_key)
        and (pt_tuple_key[-1] == "bias")
        and (pt_tuple_key[:-1] + ("bias",) not in random_flax_state_dict)
        and (pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict)
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor
    elif (
        pt_tuple_key[-1] in ["weight", "gamma"] 
        and pt_tuple_key[:-1] + ("scale",) in random_flax_state_dict
    ):
        renamed_pt_tuple_key = pt_tuple_key[:-1] + ("scale",)
        return renamed_pt_tuple_key, pt_tensor

    # embedding
    if (
        pt_tuple_key[-1] == "weight" 
        and pt_tuple_key[:-1] + ("embedding",) in random_flax_state_dict
    ):
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
        flax_key, flax_tensor = rename_key_and_reshape_tensor(
            pt_tuple_key, pt_tensor, random_flax_state_dict
        )

        if flax_key in random_flax_state_dict:
            if flax_tensor.shape != random_flax_state_dict[flax_key].shape:
                raise ValueError(
                    f'PyTorch checkpoint seems to be incorrect. Weight {pt_key} was expected '
                    f'be of shape {random_flax_state_dict[flax_key].shape}, but is '
                    f'{flax_tensor.shape}.'
                )
        # also add unexpected weight so that warning is thrown
        flax_state_dict[flax_key] = np.asarray(flax_tensor)
    return unflatten_dict(flax_state_dict)

def main():
    config_path = 'configs/stable_diffusion_2_unet.json'
    model_path = 'data/models/stable_diffusion_2_unet.safetensors'
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(json.dumps(config, indent=4))
    
    # TODO: some config settings are hardcoded and need to be made into parameters.
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

    # The stable diffusion 2 weights are distributed in torch format, so they need to be loaded
    # in torch mode then converted to flax.
    with open(model_path, 'rb') as f:
        model_weights = f.read()
    model_weights = load(model_weights)
    model_weights = convert_pytorch_state_dict_to_flax(model_weights, model)
    print(model_weights.keys())
    #with safe_open(model_path, framework='pt', device='cpu') as f:
    #    model_weights = f.read()
    #print(model_weights.keys())

if __name__ == '__main__':
    main()