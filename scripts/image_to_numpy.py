import argparse
import os
from PIL import Image
import jax.numpy as jnp

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
        image_array = jnp.array(image)
        jnp.save(os.path.join(args.output_path, f'{i}.npy'), image_array)
        image.close()
        print(f'Converted {file_name}')

if __name__ == '__main__':
    main()