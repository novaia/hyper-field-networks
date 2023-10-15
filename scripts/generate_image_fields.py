import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import json
from fields import image_field

if __name__ == '__main__':
    num_generations = 2000
    train_steps = 1000
    batch_size = 32
    mlp_width = 64
    image_directory = 'data/anime_faces'
    image_paths = os.listdir(image_directory)
    save_directory = 'data/approximation_field'
    for i in range(num_generations):
        image = Image.open(os.path.join(image_directory, image_paths[i]))
        image = jnp.array(image) / 255.0
        model = image_field.ImageField(mlp_depth=3, mlp_width=mlp_width, encoding_dim=10)
        state = image_field.create_train_state(model, 1e-3, jax.random.PRNGKey(0))
        state = image_field.train_loop(
            image, image.shape[0], image.shape[1], batch_size, train_steps, state
        )
        image_field.draw_image(
            state, image.shape[0], image.shape[1], 
            os.path.join(save_directory, f'{i}_image.png')
        )
        packed_height = 0
        packed_width = mlp_width
        weight_map = []
        packed_weights = []
        for key in state.params.keys():
            new_weight_map_entry = {'layer': key}
            for sub_key in state.params[key].keys():
                parameter = state.params[key][sub_key]
                parameter_shape = parameter.shape
                if sub_key == 'bias':
                    new_weight_map_entry['bias_width'] = parameter_shape[0]
                    packed_weights.append(jnp.expand_dims(
                        jnp.concatenate([
                            parameter, jnp.zeros((packed_width - parameter_shape[0]))
                        ], axis=-1),
                        axis=0
                    ))
                    packed_height += 1
                elif sub_key == 'kernel':
                    if parameter_shape[0] > parameter_shape[1]:
                        kernel_height = parameter_shape[1]
                        kernel_width = parameter_shape[0]
                        packed_weights.append(jnp.transpose(parameter))
                        new_weight_map_entry['kernel_transposed'] = True
                        packed_height += kernel_height
                    else:
                        kernel_height = parameter_shape[0]
                        kernel_width = parameter_shape[1]
                        packed_weights.append(parameter)
                        new_weight_map_entry['kernel_transposed'] = False
                        packed_height += kernel_height
                    new_weight_map_entry['kernel_height'] = kernel_height
                    new_weight_map_entry['kernel_width'] = kernel_width
            weight_map.append(new_weight_map_entry)
        
        packed_weights = jnp.concatenate(packed_weights, axis=0)
        packed_weights_image = packed_weights - jnp.min(packed_weights)
        packed_weights_image = packed_weights_image / jnp.max(packed_weights_image)
        plt.imsave(
            os.path.join(save_directory, f'{i}_weights.png'), 
            packed_weights_image, cmap='magma'
        )
        jnp.save(os.path.join(save_directory, f'{i}_weights.npy'), packed_weights)

    with open(os.path.join(save_directory, f'{i}_weight_map.json'), 'w') as f:
        json.dump(weight_map, f, indent=4)
    