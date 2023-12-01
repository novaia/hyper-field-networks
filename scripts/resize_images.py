import argparse
import os
from PIL import Image
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--input_extension', type=str, default='jpg')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    input_list = os.listdir(args.input_path)
    for i, file_name in enumerate(input_list):
        if not file_name.endswith('.' + args.input_extension):
            continue
        image = Image.open(os.path.join(args.input_path, file_name))
        image_array = jnp.array(image) / 255.0
        if len(image_array.shape) == 2:
            image_array = jnp.repeat(jnp.expand_dims(image_array, axis=-1), 3, axis=-1)
        image_array = jax.image.resize(image_array, (64, 64, 3), method='bilinear')
        image_array = jnp.clip(image_array, 0.0, 1.0)
        new_file_name = file_name.split('.')[0] + '.png'
        plt.imsave(os.path.join(args.output_path, new_file_name), image_array)
        image.close()
        print(f'Converted {file_name}')

if __name__ == '__main__':
    main()