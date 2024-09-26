import argparse, json, functools
import jax
from jax import numpy as jnp
from fields.ngp_image import create_model_from_config, create_train_state, render_image
from fields.common.flattening import generate_param_map, flatten_params, unflatten_params
import fp_tokenization as fpt
from matplotlib import pyplot as plt

# python -m fields.util_scripts.check_detokenized_field --field data/image_field.npy --config configs/ngp_image_robot_benchmark.json --image_width 1920 --image_height 1920

def move_pytree_to_gpu(pytree, gpu_id=0):
    device = jax.devices('gpu')[0]
    def move_to_gpu(tensor):
        return jax.device_put(jnp.array(tensor), device=device)
    return jax.tree_map(move_to_gpu, pytree)

def save_render(path:str, rendered_image:jax.Array):
    final_image = jnp.array(rendered_image, dtype=jnp.float32)
    final_image = jnp.clip(final_image, 0.0, 1.0)
    plt.imsave(path, final_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--field', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--image_height', type=int, required=True)
    parser.add_argument('--image_width', type=int, required=True)
    parser.add_argument('--u8', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.u8:
        tokenize_fn = fpt.u8_tokenize
        detokenize_fn = fpt.u8_detokenize
    else:
        tokenize_fn = fpt.tokenize
        detokenize_fn = fpt.detokenize

    with open(args.config, 'r') as f:
        config = json.load(f)
    assert(config is not None)
    print(config)

    params_cpu = dict(jnp.load(args.field, allow_pickle=True).tolist())
    #print(params_cpu)
    model = create_model_from_config(config)
    state = create_train_state(model, learning_rate=1e-3, KEY=jax.random.PRNGKey(0))
    render_fn = functools.partial(
        render_image,
        image_height=args.image_height, 
        image_width=args.image_width, 
        channels=config['channels']
    )

    # Render field with initial params.
    initial_params = move_pytree_to_gpu(params_cpu)
    state = state.replace(params=initial_params)
    initial_params_render = render_fn(state=state)

    # Render field with detokenized params.
    param_map, num_params = generate_param_map(module=initial_params)
    flattened_params = flatten_params(module=initial_params, param_map=param_map, num_params=num_params)
    tokenized_params = tokenize_fn(flattened_params)
    print(tokenized_params)
    detokenized_params = unflatten_params(
        flat_params=detokenize_fn(tokenized_params), 
        param_map=param_map
    )
    state = state.replace(params=detokenized_params)
    detokenized_params_render = render_fn(state=state) 
    
    mse = jnp.mean((detokenized_params_render - initial_params_render)**2)
    print('MSE between initial_params_render and detokenized_params_render:', mse)

    save_render(path='data/initial_field_render.png', rendered_image=initial_params_render)
    save_render(path='data/detokenized_field_render.png', rendered_image=detokenized_params_render)

if __name__ == '__main__':
    main()
