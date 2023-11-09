import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import json
from fields import image_field
from hypernets.packing.image_field import pack_weights
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_generations', type=int, default=21551)
    parser.add_argument('--train_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mlp_width', type=int, default=64)
    parser.add_argument('--mlp_depth', type=int, default=3)
    parser.add_argument('--encoding_dim', type=int, default=10)
    parser.add_argument('--image_directory', type=str, default='data/anime_faces')
    parser.add_argument('--save_directory', type=str, default='data/approximation_field')
    parser.add_argument('--preview_only', type=bool, default=False)
    args = parser.parse_args()

    model = image_field.ImageField(
        mlp_depth=args.mlp_depth, mlp_width=args.mlp_width, encoding_dim=args.encoding_dim
    )
    state = image_field.create_train_state(model, 1e-3, jax.random.PRNGKey(0))
    packed_weights, weight_map = pack_weights(state, args.mlp_width)
    print('Packed weights shape:', packed_weights.shape)
    print('Flattened weights shape:', jnp.ravel(packed_weights).shape)
    with open(os.path.join(args.save_directory, f'weight_map.json'), 'w') as f:
        json.dump(weight_map, f, indent=4)
    if args.preview_only:
        exit(0)

    image_paths = os.listdir(args.image_directory)
    for i in range(args.num_generations):
        print(f'Generation {i}...')
        image = Image.open(os.path.join(args.image_directory, image_paths[i]))
        image = jnp.array(image) / 255.0
        model = image_field.ImageField(
            mlp_depth=args.mlp_depth, mlp_width=args.mlp_width, encoding_dim=args.encoding_dim
        )
        state = image_field.create_train_state(model, 1e-3, jax.random.PRNGKey(0))
        state = image_field.train_loop(
            image, image.shape[0], image.shape[1], args.batch_size, args.train_steps, state
        )
        image_field.draw_image(
            state, image.shape[0], image.shape[1], 
            os.path.join(args.save_directory, f'{i}_image.png')
        )

        packed_weights, _ = pack_weights(state, args.mlp_width)
        packed_weights_image = packed_weights - jnp.min(packed_weights)
        packed_weights_image = packed_weights_image / jnp.max(packed_weights_image)
        plt.imsave(
            os.path.join(args.save_directory, f'{i}_weights.png'), 
            packed_weights_image, cmap='magma'
        )
        jnp.save(os.path.join(args.save_directory, f'{i}_weights.npy'), packed_weights)
    