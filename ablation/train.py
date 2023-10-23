import os
import sys
sys.path.append(os.getcwd())
import jax
import jax.numpy as jnp
from fields import ngp_nerf_cuda, Dataset, temp
from utils.types import NeRFState, OccupancyDensityGrid, StateOptions
from utils.cuda import render_rays_train
from utils import data
import optax
from functools import partial
from utils import models
from volrendjax import morton3d_invert
from utils import occupancy

def train():
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = True

    learning_rate = 1e-2
    batch_size = 256 * 1024
    scene_bound = 1.0
    grid_resolution = 128
    grid_update_interval = 16
    grid_warmup_steps = 256
    diagonal_n_steps = 1024
    train_steps = 1000
    stepsize_portion = 1.0 / 256.0

    cascades = 1
    grid_resolution = 128 
    diagonal_n_steps = 1024
    bg = False
    scene_bound = 1.0
    
    dataset = ngp_nerf_cuda.load_dataset('data/lego', 1)

    ogrid = OccupancyDensityGrid.create(cascades=cascades, grid_resolution=grid_resolution)
    # ngp_nerf_cuda model
    #'''
    nerf_model = ngp_nerf_cuda.NGPNerf(
        number_of_grid_levels=num_hash_table_levels,
        max_hash_table_entries=max_hash_table_entries,
        hash_table_feature_dim=hash_table_feature_dim,
        coarsest_resolution=coarsest_resolution,
        finest_resolution=finest_resolution,
        density_mlp_width=density_mlp_width,
        color_mlp_width=color_mlp_width,
        high_dynamic_range=high_dynamic_range,
        exponential_density_activation=exponential_density_activation,
        scene_bound=scene_bound
    )
    nerf_variables = nerf_model.init(jax.random.PRNGKey(0), (jnp.ones([3]), jnp.ones([3])))
    #'''

    # jaxngp model
    '''
    nerf_model = models.make_nerf_ngp(bound=scene_bound, tv_scale=0.0)
    nerf_variables = nerf_model.init(jax.random.PRNGKey(0), jnp.ones([1, 3]), jnp.ones([1, 3]))
    '''
    
    state_options = StateOptions(
        diagonal_n_steps=diagonal_n_steps,
        density_grid_res=grid_resolution,
        stepsize_portion=stepsize_portion,
        scene_bound=scene_bound,
        camera_cx=dataset.cx,
        camera_cy=dataset.cy,
        camera_fx=dataset.fl_x,
        camera_fy=dataset.fl_y,
        camera_width=dataset.w,
        camera_height=dataset.h,
        num_images=dataset.images.shape[0],
        cascades=cascades,
    )

    state = NeRFState.create(
        ogrid=ogrid,
        images=dataset.images,
        transform_matrices=dataset.transform_matrices,
        options=state_options,
        bg=bg,
        params=nerf_variables['params'],
        tx=make_optimizer(learning_rate, nerf_variables['params']),
        nerf_fn=nerf_model.apply,
        bg_fn=None
    )
    state = train_loop(state, train_steps, batch_size)
    jnp.save('data/occupancy_grid_density.npy', state.ogrid.occ_mask.astype(jnp.float32))
    occupancy_grid_coordinates = morton3d_invert(
        jnp.arange(state.ogrid.occ_mask.shape[0], dtype=jnp.uint32)
    )
    occupancy_grid_coordinates = occupancy_grid_coordinates / (128 - 1) * 2 - 1
    jnp.save('data/occupancy_grid_coordinates.npy', occupancy_grid_coordinates)

def make_optimizer(lr: float, params) -> optax.GradientTransformation:
    learning_rate_schedule = optax.exponential_decay(
        init_value=lr, 
        transition_steps=10_000, 
        decay_rate=1/3, 
        staircase=True,
        transition_begin=10_000,
        end_value=lr/100
    )
    adam = optax.adam(learning_rate_schedule, eps=1e-15, eps_root=1e-15)
    # To prevent divergence after long training periods, the paper applies a weak 
    # L2 regularization to the network weights, but not the hash table entries.
    weight_decay_mask = dict({
        key:True if key != 'MultiResolutionHashEncoding_0' else False
        for key in params.keys()
    })
    weight_decay = optax.add_decayed_weights(1e-6, mask=weight_decay_mask)
    return optax.chain(adam, weight_decay)

def train_loop(state:NeRFState, train_steps, batch_size):
    KEY = jax.random.PRNGKey(0)
    for step in range(train_steps):
        KEY, key_perm, key_train_step = jax.random.split(KEY, 3)
        loss, state, batch_metrics = train_step(state, key_train_step, batch_size)
        print('Step', step, 'Loss', loss, 'Num Rays', batch_metrics['n_valid_rays'])

        if state.should_call_update_ogrid:
            for cas in range(state.options.cascades):
                KEY, update_key = jax.random.split(KEY, 2)
                # vrj_test update
                new_density = jax.lax.stop_gradient(
                    occupancy.update_occupancy_grid_density(
                        KEY=update_key,
                        cas=0,
                        update_all=bool(state.should_update_all_ogrid_cells),
                        max_inference=batch_size,
                        occ_mask=state.ogrid.occ_mask,
                        alive_indices=jnp.arange(state.ogrid.occ_mask.shape[0], dtype=jnp.uint32),
                        density=state.ogrid.density,
                        density_grid_res=state.options.density_grid_res,
                        alive_indices_offset=jnp.array([0, state.ogrid.occ_mask.shape[0]], dtype=jnp.uint32),
                        scene_bound=state.options.scene_bound,
                        state=state
                    )
                )
                new_ogrid = state.ogrid.replace(density=new_density)
                state = state.replace(ogrid=new_ogrid)
                # jaxngp update
                '''
                state = state.update_ogrid_density(
                    KEY=update_key,
                    cas=cas,
                    update_all=bool(state.should_update_all_ogrid_cells),
                    max_inference=batch_size,
                )
                '''
            state = state.threshold_ogrid()
    return state

@partial(jax.jit, static_argnames=('total_samples'))
def train_step(state, KEY, total_samples):
    KEY, pixel_sample_key = jax.random.split(KEY, 2)
    image_indices, width_indices, height_indices = ngp_nerf_cuda.sample_pixels(
        key=pixel_sample_key, 
        num_samples=total_samples,
        image_width=state.options.camera_width, 
        image_height=state.options.camera_height, 
        num_images=state.options.num_images, 
    )
    get_rays = jax.vmap(ngp_nerf_cuda.get_ray, in_axes=(0, 0, 0, None, None, None, None))
    ray_origins, ray_directions = get_rays(
        width_indices, height_indices, state.transform_matrices[image_indices],
        state.options.camera_cx, state.options.camera_cy, state.options.camera_fx, 
        state.options.camera_fy
    )

    def loss_fn(params, gt_rgba, KEY):
        KEY, bg_key = jax.random.split(KEY, 2)
        bg = jax.random.uniform(
            key=bg_key, shape=(total_samples, 3), 
            dtype=jnp.float32, minval=0.0, maxval=1.0
        )

        KEY, render_key = jax.random.split(KEY, 2)
        batch_metrics, pred_rgbds = render_rays_train(
            KEY=render_key,
            o_world=ray_origins,
            d_world=ray_directions,
            bg=bg,
            total_samples=total_samples,
            # This replace is very important.
            # Without it the params aren't optimized.
            state=state.replace(params=params)
        )
        pred_rgbs, pred_depths = jnp.array_split(pred_rgbds, [3], axis=-1)
        gt_rgbs = data.blend_rgba_image_array(gt_rgba, bg)
        loss = jnp.where(
            batch_metrics["ray_is_valid"],
            optax.huber_loss(pred_rgbs, gt_rgbs, delta=0.1).mean(axis=-1),
            0.,
        ).sum() / batch_metrics["n_valid_rays"]
        return loss, batch_metrics
    
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    KEY, loss_key = jax.random.split(KEY, 2)
    gt_rgba = state.images[image_indices, height_indices, width_indices]
    (loss, batch_metrics), grads = loss_grad_fn(state.params, gt_rgba, loss_key)
    state = state.apply_gradients(grads=grads)
    return loss, state, batch_metrics

if __name__ == '__main__':
    train()