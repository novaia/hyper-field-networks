import argparse
import os
import glob
from fields import ngp_image
import jax
import jax.numpy as jnp
from PIL import Image
import copy
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from fields.common.flattening import generate_param_map, flatten_params

def get_dataset(dataset_path):
    if dataset_path.endswith('/'):
        glob_pattern = f'{dataset_path}*.parquet'
    else:
        glob_pattern = f'{dataset_path}/*.parquet'
    parquet_files = glob.glob(glob_pattern)
    assert len(parquet_files) > 0, 'No parquet files were found in dataset directory.'
    print(f'Found {len(parquet_files)} parquet files in dataset directory.')
    dataset = load_dataset(
        'parquet', 
        data_files={'train': parquet_files},
        split='train',
        num_proc=8
    )
    dataset = dataset.with_format('jax')
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--field_type', type=str, choices=['ngp_image', 'ngp_nerf'], required=True)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    channels = config['channels']

    dataset = get_dataset(args.input_path)
    dataset_iterator = dataset.iter(batch_size=1)
    num_samples = len(dataset)
    print('Samples in dataset:', num_samples)

    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)
    initial_opt_state = copy.deepcopy(state.opt_state)
    initial_tx = copy.deepcopy(state.tx)
    param_map, _ = generate_param_map(state.params)

    with open(os.path.join(args.output_path, 'param_map.json'), 'w') as f:
        json.dump(param_map, f, indent=4)

    first_image = dataset[0]['image']
    num_pixels = first_image.shape[0] * first_image.shape[1]
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f'Num pixels: {num_pixels:,}')
    print(f'Num params: {num_params:,}')
    print('Param to pixel ratio:', num_params/num_pixels)

    num_retries = 4
    loss_threshold = 6e-5
    for i in range(num_samples):
        image = next(dataset_iterator)['image'][0] / 255.0
        state = state.replace(params=initial_params, tx=initial_tx, opt_state=initial_opt_state, step=0)
        
        for k in range(num_retries):
            state = ngp_image.train_loop(
                config['train_steps'], state, image, config['batch_size']
            )
            rendered_image = ngp_image.render_image(
                state, image.shape[0], image.shape[1], channels=channels
            )
            full_image_loss = jnp.mean(image - rendered_image)**2
            print(f'Sample {i}, attempt {k}, loss: {full_image_loss}')
            if full_image_loss < loss_threshold:
                break
        flat_params = flatten_params(state.params, param_map, num_params)
        print(f'Sample {i}, param mean: {jnp.mean(flat_params)}, param var: {jnp.var(flat_params)}')
        plt.imsave(os.path.join(args.output_path, f'{i}.jpg'), rendered_image)

if __name__ == '__main__':
    main()
