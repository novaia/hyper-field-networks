import os
import glob
from fields import ngp_image
import jax
import jax.numpy as jnp
from PIL import Image
import copy
import json
import matplotlib.pyplot as plt

def main():
    with open('configs/ngp_image.json', 'r') as f:
        config = json.load(f)

    output_path = 'data/mnist_ingp'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    input_path = 'data/mnist_png/data'
    dataset_list = os.listdir(input_path)
    model = ngp_image.create_model_from_config(config)
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    initial_params = copy.deepcopy(state.params)
    
    cpu = jax.devices('cpu')[0]
    for i, path in enumerate(dataset_list):
        pil_image = Image.open(os.path.join(input_path, path))
        image = jnp.array(pil_image)
        image = jnp.array(image)/255.0
        state = state.replace(params=initial_params)
        state, final_loss = ngp_image.train_loop(
            config['train_steps'], state, image, config['batch_size'], True
        )
        print(f'Final loss: {final_loss}')
        if final_loss < 0.015:
            jnp.save(os.path.join(output_path, f'{i}.npy'), dict(state.params), allow_pickle=True)
            rendered_image = ngp_image.render_image(state, image.shape[0], image.shape[1])
            #print(rendered_image.shape)
            rendered_image = jax.device_put(rendered_image, cpu)
            plt.imsave(os.path.join(output_path, f'{i}.png'), rendered_image)
            rendered_image.delete()
        
        image.delete()
        pil_image.close()

if __name__ == '__main__':
    main()
