import json
import jax
from jax import numpy as jnp
from datasets import load_dataset
from fields.ngp_image import (
    create_model_from_config, create_train_state, render_image
)
from fields.common.flattening import unflatten_params
from matplotlib import pyplot as plt
from fp_conversion import tokenize, detokenize

with open('configs/colored_monsters_ngp_image.json', 'r') as f:
    config = json.load(f)
with open('data/something3/param_map.json', 'r') as f:
    param_map = json.load(f)

dataset = load_dataset('parquet', data_files={'train': 'data/output.parquet'}, split='train')
dataset = dataset.with_format('jax')
flat_params = dataset[40]['params']
flat_params = jnp.array(flat_params, dtype=jnp.float16)
print('flat params shape', flat_params.shape)

param_tokens = tokenize(flat_params)
print(param_tokens[:20])
flat_params = detokenize(param_tokens)

params = unflatten_params(flat_params, param_map)

model = create_model_from_config(config)
state = create_train_state(model, 1e-3, jax.random.PRNGKey(0))
state = state.replace(params=params)

image = render_image(state, 256, 256, config['channels'])
print(image)
print(image.shape)
plt.imsave('data/test_pq.png', image)
