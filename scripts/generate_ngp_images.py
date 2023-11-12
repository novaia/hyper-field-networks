import os
import sys
sys.path.append(os.getcwd())

from fields import ngp_image
import argparse
import json
import jax
import jax.numpy as jnp
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        output_path_list = []
    else:
        output_path_list = os.listdir(args.output_path)

    assert args.config.endswith('.json'), 'Config file must be a JSON file'
    assert os.path.isfile(args.config), f'Config file {args.config} does not exist'
    with open(args.config) as f:
        config = json.load(f)

    input_path_list = os.listdir(args.input_path)
    state_init_key = jax.random.PRNGKey(0)

    for path in input_path_list:
        if not path.endswith('.png') and not path.endswith('.jpg'):
            continue
        output_name = f'{path[:-4]}.npy'
        if output_name in output_path_list:
            print(f'Input file {path} already exists in output as {output_name}, skipping...')
            continue
        print(f'Generating NGP image for {path}...')
        model = ngp_image.NGPImage(
            number_of_grid_levels=config['num_hash_table_levels'],
            max_hash_table_entries=config['max_hash_table_entries'],
            hash_table_feature_dim=config['hash_table_feature_dim'],
            coarsest_resolution=config['coarsest_resolution'],
            finest_resolution=config['finest_resolution'],
            mlp_width=config['mlp_width'],
            mlp_depth=config['mlp_depth']
        )
        state = ngp_image.create_train_state(model, config['learning_rate'], state_init_key)
        pil_image = Image.open(os.path.join(args.input_path, path))
        image = jnp.array(pil_image)
        image = jnp.array(image)/255.0
        state, final_loss = ngp_image.train_loop(
            config['train_steps'], state, image, config['batch_size'], True
        )
        print(f'Final loss: {final_loss}')
        output_dict = {'final_loss': final_loss, 'params': dict(state.params)}
        jnp.save(
            os.path.join(args.output_path, output_name), 
            output_dict,
            allow_pickle=True
        )
        pil_image.close()

if __name__ == '__main__':
    main()