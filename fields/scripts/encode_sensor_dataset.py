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

def train(input_path:str, output_path:str, config:dict, render:bool):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        output_path_list = []
    else:
        output_path_list = glob.glob(f'{output_path}/*.npy')
    #input_path_list = os.listdir(input_path)
    input_path_list = glob.glob(f'{input_path}/*.png')
    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)

    for path in input_path_list:
        if not path.endswith('.png') and not path.endswith('.jpg'):
            continue
        split_path = path.split('/')
        file_name = split_path[-1]
        bucket_name = split_path[-2]
        output_bucket_path = f'{output_path}/{bucket_name}'
        output_file_name = os.path.join(output_bucket_path, f'{file_name[:-4]}.npy')
        if output_file_name in output_path_list:
            print(f'Input file {path} already exists in output as {output_file_name}, skipping...')
            continue
        print(f'Generating NGP image for {path}...')
        
        state = state.replace(params=initial_params)
        pil_image = Image.open(path)
        image = jnp.array(pil_image)
        image = jnp.array(image)/255.0
        state, final_loss = ngp_image.train_loop(
            config['train_steps'], state, image, config['batch_size'], True
        )
        print(f'Final loss: {final_loss}')
        output_dict = {'final_loss': final_loss, 'params': dict(state.params)}
        if not os.path.exists(output_bucket_path):
            os.makedirs(output_bucket_path)
        jnp.save(output_file_name, output_dict, allow_pickle=True)
        if render:
            rendered_image = ngp_image.render_image(state, image.shape[0], image.shape[1])
            plt.imsave(f'{output_file_name[:-4]}.jpg', rendered_image)
            rendered_image.delete()
        image.delete()
        del output_dict
        pil_image.close()

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
    
    print(args.field_type)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    channels = config['channels']

    dataset = get_dataset(args.input_path)
    dataset_iterator = dataset.iter(batch_size=1)
    num_samples = len(dataset)
    print(len(dataset))

    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)
    initial_opt_state = copy.deepcopy(state.opt_state)
    initial_tx = copy.deepcopy(state.tx)
    
    for i in range(num_samples):
        image = next(dataset_iterator)['image'][0] / 255.0
        state = state.replace(params=initial_params, tx=initial_tx, opt_state=initial_opt_state, step=0)
        print('sample', i)
        print(state.opt_state)
        state = ngp_image.train_loop(
            config['train_steps'], state, image, config['batch_size']
        )
        if args.render:
            rendered_image = ngp_image.render_image(
                state, image.shape[0], image.shape[1], channels=channels
            )
            plt.imsave(os.path.join(args.output_path, f'{i}.jpg'), rendered_image)

if __name__ == '__main__':
    main()
