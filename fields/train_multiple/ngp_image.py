import argparse
import os
from fields import ngp_image
import jax
import jax.numpy as jnp
from PIL import Image
import copy
import json
import matplotlib.pyplot as plt

def train(input_path:str, output_path:str, config:dict, render:bool):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        output_path_list = []
    else:
        output_path_list = os.listdir(output_path)
    input_path_list = os.listdir(input_path)
    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)

    for path in input_path_list:
        if not path.endswith('.png') and not path.endswith('.jpg'):
            continue
        output_name = f'{path[:-4]}.npy'
        if output_name in output_path_list:
            print(f'Input file {path} already exists in output as {output_name}, skipping...')
            continue
        print(f'Generating NGP image for {path}...')
        
        state = state.replace(params=initial_params)
        pil_image = Image.open(os.path.join(input_path, path))
        image = jnp.array(pil_image)
        image = jnp.array(image)/255.0
        state, final_loss = ngp_image.train_loop(
            config['train_steps'], state, image, config['batch_size'], True
        )
        print(f'Final loss: {final_loss}')
        output_dict = {'final_loss': final_loss, 'params': dict(state.params)}
        jnp.save(os.path.join(output_path, output_name), output_dict, allow_pickle=True)
        if render:
            rendered_image = ngp_image.render_image(state, image.shape[0], image.shape[1])
            plt.imsave(os.path.join(output_path, path), rendered_image)
            rendered_image.delete()
        image.delete()
        del output_dict
        pil_image.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/ngp_image.json')
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    train(args.input_path, args.output_path, config, args.render)

if __name__ == '__main__':
    main()