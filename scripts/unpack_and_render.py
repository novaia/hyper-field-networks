import os
import sys
sys.path.append(os.getcwd())
from hypernets.packing.ngp import unpack_weights
import json
import jax.numpy as jnp
import jax
from fields.ngp_image import NGPImage, create_train_state, render_image
import matplotlib.pyplot as plt

def main():
    weight_map_path = 'data/packed_ngp_anime_faces/weight_map.json'
    with open(weight_map_path) as f:
        weight_map = json.load(f)
    packed_weights_path = 'data/packed_ngp_anime_faces/11258.npy'
    packed_weights = jnp.load(packed_weights_path)
    unpacked_weights, parsed_height = unpack_weights(packed_weights, weight_map)

    with open('configs/ngp_image.json', 'r') as f:
        config = json.load(f)
    
    model = NGPImage(
        number_of_grid_levels=config['num_hash_table_levels'],
        max_hash_table_entries=config['max_hash_table_entries'],
        hash_table_feature_dim=config['hash_table_feature_dim'],
        coarsest_resolution=config['coarsest_resolution'],
        finest_resolution=config['finest_resolution'],
        mlp_width=config['mlp_width'],
        mlp_depth=config['mlp_depth']
    )
    state = create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    state = state.replace(params=unpacked_weights)
    rendered_image = render_image(state, 64, 64)
    plt.imsave('data/unpacked.png', rendered_image)

if __name__ == '__main__':
    main()