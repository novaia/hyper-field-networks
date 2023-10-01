# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

import flax.linen as nn
from flax.training.train_state import TrainState
import jax.numpy as jnp
import jax
from jax.random import PRNGKeyArray
import optax
import json
import os
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from functools import partial

@dataclass
class Dataset:
    horizontal_fov: float
    vertical_fov: float
    fl_x: float # Focal length x.
    fl_y: float # Focal length y.
    k1: float # First radial distortion parameter.
    k2: float # Second radial distortion parameter.
    p1: float # Third radial distortion parameter.
    p2: float # Fourth radial distortion parameter.
    cx: float # Principal point x.
    cy: float # Principal point y.
    w: int # Image width.
    h: int # Image height.
    aabb_scale: int # Scale of scene bounding box.
    canvas_plane: Optional[float] = 1.0 # Distance from center of projection to canvas plane.
    transform_matrices: Optional[jnp.ndarray] = None
    locations: Optional[jnp.ndarray] = None
    directions: Optional[jnp.ndarray] = None
    images: Optional[jnp.ndarray] = None
    canvas_width_ratio: Optional[float] = None
    canvas_height_ratio: Optional[float] = None

# Calculates the fourth order spherical harmonic encoding for the given direction.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
def fourth_order_sh_encoding(direction:jnp.ndarray):
    x = direction[0]
    y = direction[1]
    z = direction[2]

    xx = x**2
    yy = y**2
    zz = z**2

    components = jnp.array([
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.9461746957575601 * zz - 0.31539156525251999,
        1.0925484305920792 * x * z,
        0.5462742152960396 * (xx - yy),
        0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * x * y * z,
        0.4570457994644658 * y * (5 * zz - 1),
        0.3731763325901154 * z * (5 * zz - 3),
        0.4570457994644658 * x * (5 * zz - 1),
        1.445305721320277 * z * (xx - yy),
        0.5900435899266435 * x * (xx - 3 * yy)
    ])

    return components

def render_pixel(densities:jnp.ndarray, colors:jnp.ndarray, deltas:jnp.ndarray):    
    expanded_densities = jnp.expand_dims(densities, axis=0)
    repeated_densities = jnp.repeat(expanded_densities, densities.shape[0], axis=0)
    triangular_mask = jnp.tril(jnp.ones(repeated_densities.shape))
    print('Triangular mask shape:', triangular_mask.shape)
    print('Repeated densities shape:', repeated_densities.shape)
    triangular_densities = repeated_densities * triangular_mask
    print('Triangular densities shape:', triangular_densities.shape)
    expanded_deltas = jnp.expand_dims(deltas, axis=0)
    print('Expanded deltas shape:', expanded_deltas.shape)
    repeated_deltas = jnp.repeat(expanded_deltas, deltas.shape[0], axis=0)
    triangular_deltas = repeated_deltas * triangular_mask
    print('Triangular deltas shape:', triangular_deltas.shape)

    T_sum = jnp.exp(-jnp.sum(triangular_densities * triangular_deltas, axis=1))
    print('T_sum shape:', T_sum.shape)
    rendered_color = jnp.sum(T_sum * (1 - jnp.exp(-densities * deltas)) * colors, axis=0)
    print('Rendered color shape:', rendered_color.shape)
    return rendered_color

class MultiResolutionHashEncoding(nn.Module):
    table_init_key: PRNGKeyArray
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.spatial_dim = 3
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        self.scalings = jnp.reshape(self.scalings, (self.scalings.shape[0], 1))
        absolute_table_size = self.table_size * self.num_levels
        # Feature dim comes first so features can be broadcast with point offset.
        # Feature shape is (feature_dim, num_levels). 
        # Point offset shape is (spatial_dim, num_levels).
        self.hash_table = jax.random.normal(
            self.table_init_key, 
            shape=(self.feature_dim, absolute_table_size)
        )

    def hash_function(self, x:jnp.ndarray, table_size:int, hash_offset:jnp.ndarray):
        pre_xor = x * jnp.array([1, 2654435761, 805459861])
        x = jnp.bitwise_xor(pre_xor[:, 0], pre_xor[:, 1])
        x = jnp.bitwise_xor(x, pre_xor[:, 2])
        x %= table_size
        x += hash_offset
        return x
    
    def __call__(self, x:jnp.ndarray):
        scaled = x * self.scalings
        scaled_c = jnp.ceil(scaled).astype(jnp.int32)
        scaled_f = jnp.floor(scaled).astype(jnp.int32)
        point_offset = jnp.reshape(scaled - scaled_f, (self.spatial_dim, self.num_levels))
        
        vertex_0 = scaled_c
        vertex_1 = jnp.concatenate([scaled_c[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_2 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_3 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_4 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_5 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_6 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_7 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)

        hashed_0 = self.hash_function(vertex_0, self.table_size, self.hash_offset)
        hashed_1 = self.hash_function(vertex_1, self.table_size, self.hash_offset)
        hashed_2 = self.hash_function(vertex_2, self.table_size, self.hash_offset)
        hashed_3 = self.hash_function(vertex_3, self.table_size, self.hash_offset)
        hashed_4 = self.hash_function(vertex_4, self.table_size, self.hash_offset)
        hashed_5 = self.hash_function(vertex_5, self.table_size, self.hash_offset)
        hashed_6 = self.hash_function(vertex_6, self.table_size, self.hash_offset)
        hashed_7 = self.hash_function(vertex_7, self.table_size, self.hash_offset)

        f_0 = self.hash_table[:, hashed_0]
        f_1 = self.hash_table[:, hashed_1]
        f_2 = self.hash_table[:, hashed_2]
        f_3 = self.hash_table[:, hashed_3]
        f_4 = self.hash_table[:, hashed_4]
        f_5 = self.hash_table[:, hashed_5]
        f_6 = self.hash_table[:, hashed_6]
        f_7 = self.hash_table[:, hashed_7]

        # Linearly interpolate between all of the features.
        f_03 = f_0 * point_offset[0:1, :] + f_3 * (1 - point_offset[0:1, :])
        f_12 = f_1 * point_offset[0:1, :] + f_2 * (1 - point_offset[0:1, :])
        f_56 = f_5 * point_offset[0:1, :] + f_6 * (1 - point_offset[0:1, :])
        f_47 = f_4 * point_offset[0:1, :] + f_7 * (1 - point_offset[0:1, :])

        f0312 = f_03 * point_offset[1:2, :] + f_12 * (1 - point_offset[1:2, :])
        f4756 = f_47 * point_offset[1:2, :] + f_56 * (1 - point_offset[1:2, :])

        encoded_value = f0312 * point_offset[2:3, :] + f4756 * (
            1 - point_offset[2:3, :]
        )
        # Transpose so that features are contiguous.
        # i.e. [[f_0_x, f_0_y], [f_1_x, f_2_y], ...]
        # Then ravel to get the entire encoding.
        return jnp.ravel(jnp.transpose(encoded_value))

class InstantNerf(nn.Module):
    hash_table_init_rng: PRNGKeyArray
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.
    density_mlp_width: int
    color_mlp_width: int
    high_dynamic_range: bool

    @nn.compact
    def __call__(self, x):
        position, direction = x
        encoded_position = MultiResolutionHashEncoding(
            table_init_key=self.hash_table_init_rng,
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim
        )(position)

        x = nn.Dense(self.density_mlp_width)(encoded_position)
        x = nn.activation.relu(x)
        density_output = nn.Dense(16)(x)
        density = density_output[0]

        encoded_direction = fourth_order_sh_encoding(direction)
        # Encoded_direction is currently 3x16 but I'm not sure if it is supposed to be.
        # For now I'm just going to ravel it to 48x1 so it can be concatenated with density.
        x = jnp.concatenate([density_output, jnp.ravel(encoded_direction)], axis=0)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(self.color_mlp_width)(x)
        x = nn.activation.relu(x)
        x = nn.Dense(3)(x)

        if self.high_dynamic_range:
            # elu is exponential linear unit, I think that's what the paper meant 
            # by "exponential activation"
            color = nn.activation.elu(x)
        else:
            color = nn.activation.sigmoid(x)

        return density, color

def create_train_state(model:nn.Module, rng:PRNGKeyArray, learning_rate:float):
    x = (jnp.ones([3]) / 3, jnp.ones([3]) / 3)
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate)
    ts = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return ts

def sample_pixels(
    num_samples:int, 
    image_width:int, 
    image_height:int, 
    num_images:int, 
    images:jnp.ndarray,
    rng:PRNGKeyArray
):
    width_rng, height_rng, image_rng = jax.random.split(rng, num=3) 
    width_indices = jax.random.randint(
        width_rng, shape=(num_samples,), minval=0, maxval=image_width
    )
    height_indices = jax.random.randint(
        height_rng, shape=(num_samples,), minval=0, maxval=image_height
    )
    image_indices = jax.random.randint(
        image_rng, shape=(num_samples,), minval=0, maxval=num_images
    )
    pixel_samples = images[image_indices, width_indices, height_indices]
    indices = (image_indices, width_indices, height_indices)
    return pixel_samples, indices 

def get_ray_scales(ray_near:float, ray_far:float, batch_size:int, num_samples:int):
    ray_scales = jnp.linspace(ray_near, ray_far, num_samples)
    # (1, num_ray_samples, 1)
    ray_scales = jnp.expand_dims(ray_scales, axis=(0, -1))
    # (batch_size, num_ray_samples, 3)
    ray_scales = jnp.repeat(jnp.repeat(ray_scales, 3, axis=-1), batch_size, axis=0)
    return ray_scales

def train_loop(batch_size:int, training_steps:int, state:TrainState, dataset:Dataset):
    num_ray_samples = 64
    ray_scales = get_ray_scales(
        ray_near=dataset.canvas_plane, 
        ray_far=40.0, 
        batch_size=batch_size, 
        num_samples=num_ray_samples
    )

    for step in range(training_steps):
        rng = jax.random.PRNGKey(step)
        loss, state = train_step(
            batch_size=batch_size,
            image_width=dataset.w,
            image_height=dataset.h,
            canvas_width_ratio=dataset.canvas_width_ratio,
            canvas_height_ratio=dataset.canvas_height_ratio,
            canvas_plane=dataset.canvas_plane,
            transform_matrices=dataset.transform_matrices,
            images=dataset.images,
            state=state, 
            num_ray_samples=num_ray_samples, 
            ray_scales=ray_scales, 
            rng=rng
        )
        print('Loss:', loss)

@partial(jax.jit, static_argnames=(
    'batch_size',
    'image_width',
    'image_height',
    'num_ray_samples',
))
def train_step(
    batch_size:int,
    image_width:int,
    image_height:int,
    canvas_width_ratio:float,
    canvas_height_ratio:float,
    canvas_plane:float,
    transform_matrices:jnp.ndarray,
    images:jnp.ndarray,
    state:TrainState,
    num_ray_samples:int,
    ray_scales:jnp.ndarray,
    rng:PRNGKeyArray
):
    source_pixels, indices = sample_pixels(
        num_samples=batch_size, 
        image_width=image_width, 
        image_height=image_height, 
        num_images=images.shape[0], 
        rng=rng, 
        images=images
    )

    image_indices, width_indices, height_indices = indices
    # Scale from real canvas dimensions to virtual canvas dimensions.
    rays_x = width_indices * canvas_width_ratio
    rays_y = height_indices * canvas_height_ratio
    rays_z = jnp.repeat(jnp.array([canvas_plane]), rays_x.shape[0])

    # Repeat rays along a new axis and then scale them to get samples at different points.
    rays = jnp.stack([rays_x, rays_y, rays_z], axis=-1)
    rays = jnp.repeat(jnp.expand_dims(rays, axis=1), num_ray_samples, axis=1)
    rays = rays * ray_scales

    # Convert rays to homogenous coordinates by adding w = 1 component.
    rays_w = jnp.ones((num_ray_samples, 1))
    concat_w = jax.vmap(lambda r, w: jnp.concatenate([r, w], axis=-1), in_axes=(0, None))
    rays = concat_w(rays, rays_w)

    # Transform rays from camera space to world space.
    selected_transform_matrices = transform_matrices[image_indices]
    # Map both inputs over batch dimenension, then map rays over sample dimension.
    transform_rays = jax.vmap(jax.vmap(lambda t, r: t @ r, in_axes=(None, 0)), in_axes=0)
    rays = transform_rays(selected_transform_matrices, rays)
    # Convert rays back to Cartesian coordinates.
    rays = rays[:, :, :3]

    # Ray origins can be extracted from transform matrices without matmul.
    # They are equal to the translation components of the transform matrices.
    ray_origins = selected_transform_matrices[:, :3, 3]
    ray_origins = jnp.expand_dims(ray_origins, axis=1)
    rays_with_origins = jnp.concatenate([ray_origins, rays], axis=1)

    directions = rays[:, 1] - rays[:, 0]
    directions = jnp.repeat(jnp.expand_dims(directions, axis=1), num_ray_samples, axis=1)

    def get_output(params, rays, directions):
        return state.apply_fn({'params': params}, (rays, directions))
    get_output_batch_vmap = jax.vmap(get_output, in_axes=(None, 0, 0))
    get_output_sample_vmap = jax.vmap(get_output_batch_vmap, in_axes=(None, 0, 0))

    def get_rendered_pixel(densities, colors, rays_with_origins):
        vector_deltas = jnp.diff(rays_with_origins, axis=0)
        deltas = jnp.sqrt(
            vector_deltas[:, 0]**2 + vector_deltas[:, 1]**2 + vector_deltas[:, 2]**2
        )
        deltas = jnp.expand_dims(deltas, axis=-1)
        return render_pixel(densities, colors, deltas)
    get_rendered_pixel_vmap = jax.vmap(get_rendered_pixel, in_axes=0)

    def loss_fn(params):
        densities, colors = get_output_sample_vmap(params, rays, directions)
        densities = jnp.expand_dims(densities, axis=-1)
        rendered_pixels = get_rendered_pixel_vmap(densities, colors, rays_with_origins)
        loss = jnp.mean((rendered_pixels - source_pixels[:, :3])**2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def load_dataset(path:str):
    with open(os.path.join(path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        image = Image.open(os.path.join(path, frame['file_path']))
        images.append(jnp.array(image))

    dataset = Dataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_y'],
        fl_x=transforms['fl_x'],
        fl_y=transforms['fl_y'],
        k1=transforms['k1'],
        k2=transforms['k2'],
        p1=transforms['p1'],
        p2=transforms['p2'],
        cx=transforms['cx'],
        cy=transforms['cy'],
        w=transforms['w'],
        h=transforms['h'],
        aabb_scale=transforms['aabb_scale'],
        canvas_plane=1.0,
        transform_matrices=jnp.array(transform_matrices),
        images=jnp.array(images) / 255.0
    )

    virtual_canvas_x = dataset.canvas_plane * jnp.tan(dataset.horizontal_fov/2)
    virtual_canvas_y = dataset.canvas_plane * jnp.tan(dataset.vertical_fov/2)
    real_canvas_x = dataset.cx
    real_canvas_y = dataset.cy
    dataset.canvas_width_ratio = virtual_canvas_x / real_canvas_x
    dataset.canvas_height_ratio = virtual_canvas_y / real_canvas_y

    return dataset

if __name__ == '__main__':
    print('GPU:', jax.devices('gpu'))

    dataset_path = 'data/generation_0'
    dataset = load_dataset(dataset_path)
    print(dataset.horizontal_fov)
    print(dataset.vertical_fov)
    print(dataset.fl_x)
    print(dataset.fl_y)
    print(dataset.k1)
    print(dataset.k2)
    print(dataset.p1)
    print(dataset.p2)
    print(dataset.cx)
    print(dataset.cy)
    print(dataset.w)
    print(dataset.h)
    print(dataset.aabb_scale)
    print('Images shape:', dataset.images.shape)

    model = InstantNerf(
        hash_table_init_rng=jax.random.PRNGKey(0),
        number_of_grid_levels=16,
        max_hash_table_entries=2**14,
        hash_table_feature_dim=2,
        coarsest_resolution=16,
        finest_resolution=1024,
        density_mlp_width=64,
        color_mlp_width=64,
        high_dynamic_range=False
    )
    rng = jax.random.PRNGKey(1)
    state = create_train_state(model, rng, 1e-4)
    train_loop(
        batch_size=10000, 
        training_steps=100, 
        state=state, 
        dataset=dataset
    )