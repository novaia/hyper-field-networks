from fields import ngp_image, image_field
from hypernets.packing.ngp import unpack_weights
import jax
import json

def unpack_and_render_ngp_image(
    config_path:str, weight_map_path:str, packed_weights:jax.Array, 
    image_width:int, image_height:int
):
    with open(weight_map_path, 'r') as f:
        weight_map = json.load(f)
    unpacked_weights = unpack_weights(packed_weights, weight_map)[0]

    with open(config_path, 'r') as f:
        config = json.load(f)
    model = ngp_image.NGPImage(
        number_of_grid_levels=config['num_hash_table_levels'],
        max_hash_table_entries=config['max_hash_table_entries'],
        hash_table_feature_dim=config['hash_table_feature_dim'],
        coarsest_resolution=config['coarsest_resolution'],
        finest_resolution=config['finest_resolution'],
        mlp_width=config['mlp_width'],
        mlp_depth=config['mlp_depth']
    )
    state = ngp_image.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    state = state.replace(params=unpacked_weights)
    rendered_image = ngp_image.render_image(state, image_height, image_width)
    del state, model, unpacked_weights, weight_map, config
    return rendered_image

def unpack_and_render_image_field(
    config_path:str, weight_map_path:str, packed_weights:jax.Array, 
    image_width:int, image_height:int
):
    with open(weight_map_path, 'r') as f:
        weight_map = json.load(f)
    unpacked_weights = unpack_weights(packed_weights, weight_map)[0]

    with open(config_path, 'r') as f:
        config = json.load(f)
    model = image_field.create_model_from_config(config)
    state = image_field.create_train_state(model, config['learning_rate'], jax.random.PRNGKey(0))
    state = state.replace(params=unpacked_weights)
    rendered_image = image_field.render_image(state, image_height, image_width)
    del state, model, unpacked_weights, weight_map, config
    return rendered_image
