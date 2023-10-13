import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from dataclasses import dataclass
from jax.random import PRNGKeyArray
import optax
import json
import os
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Dataset:
    horizontal_fov: float
    vertical_fov: float
    fl_x: float # Focal length x.
    fl_y: float # Focal length y.
    cx: float # Principal point x.
    cy: float # Principal point y.
    w: int # Image width.
    h: int # Image height.
    transform_matrices: Optional[jnp.ndarray] = None
    images: Optional[jnp.ndarray] = None

def render_pixel(
    densities:jnp.ndarray, colors:jnp.ndarray, z_vals:jnp.ndarray, directions:jnp.ndarray
):  
    eps = 1e-10
    deltas = jnp.concatenate([
        z_vals[1:] - z_vals[:-1], 
        jnp.broadcast_to(1e10, z_vals[:1].shape)
    ], axis=-1)
    deltas = jnp.expand_dims(deltas, axis=-1) * jnp.linalg.norm(directions, axis=-1)
    alphas = 1.0 - jnp.exp(-densities * deltas)
    accum_prod = jnp.concatenate([
        jnp.ones_like(alphas[:1], alphas.dtype),
        jnp.cumprod(1.0 - alphas[:-1] + eps, axis=0)
    ], axis=0)
    weights = alphas * accum_prod
    rendered_color = jnp.sum(weights * colors, axis=0)
    accumulated_alpha = jnp.sum(weights, axis=0)
    return rendered_color, accumulated_alpha

def get_ray_samples(
    uv_x, 
    uv_y, 
    transform_matrix, 
    rng, 
    c_x, 
    c_y, 
    fl_x, 
    fl_y, 
    ray_near, 
    ray_far, 
    num_ray_samples, 
    randomize=False
):
    direction = jnp.array([(uv_x - c_x) / fl_x, -(uv_y - c_y) / fl_y, -1.0])
    direction = transform_matrix[:3, :3] @ direction
    origin = transform_matrix[:3, -1]

    t_vals = jnp.linspace(0., 1., num_ray_samples)
    z_vals = ray_near * (1. - t_vals) + ray_far * t_vals

    if randomize:
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = jnp.concatenate([mids, z_vals[-1:]], -1)
        lower = jnp.concatenate([z_vals[:1], mids], -1)
        t_rand = jax.random.uniform(rng, (num_ray_samples,))
        z_vals = lower + (upper - lower) * t_rand

    repeated_directions = jnp.repeat(jnp.expand_dims(direction, axis=0), num_ray_samples, axis=0)
    expanded_zvals = jnp.expand_dims(z_vals, axis=-1)
    ray_samples = origin + repeated_directions * expanded_zvals
    return ray_samples, direction, z_vals

def frequency_encoding(x, min_deg, max_deg):
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    xb = x * scales
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.ravel(jnp.concatenate([x] + [four_feat], axis=-1))

class VanillaNerf(nn.Module):
    density_mlp_width: int
    density_mlp_depth: int
    color_mlp_width: int
    color_mlp_depth: int
    exponential_density_activation: bool
    positional_encoding_dim: int
    directional_encoding_dim: int

    @nn.compact
    def __call__(self, x):
        position, direction = x
        encoded_position = frequency_encoding(
            jnp.expand_dims(position, axis=-1), 
            0, 
            self.positional_encoding_dim
        )

        x = encoded_position
        for i in range(self.density_mlp_depth):
            if i % 4 == 0 and i > 0:
                x = jnp.concatenate([x, encoded_position], axis=0)
            x = nn.Dense(self.density_mlp_width)(x)
            x = nn.activation.relu(x)
        x = nn.Dense(self.density_mlp_width+1)(x)
        if self.exponential_density_activation:
            density = jnp.exp(x[0:1])
        else:
            density = nn.activation.relu(x[0:1])
        density_feature = x

        encoded_direction = frequency_encoding(
            jnp.expand_dims(direction, axis=-1),
            0,
            self.directional_encoding_dim
        )
        x = jnp.concatenate([density_feature, encoded_direction], axis=-1)
        for i in range(self.color_mlp_depth):
            x = nn.Dense(self.color_mlp_width)(x)
            x = nn.activation.relu(x)
        x = nn.Dense(3)(x)
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

def train_loop(
    batch_size:int, 
    num_ray_samples:int,
    ray_near:float,
    ray_far:float, 
    training_steps:int, 
    state:TrainState, 
    dataset:Dataset
):
    for step in range(training_steps):
        rng = jax.random.PRNGKey(step)
        loss, state = train_step(
            batch_size=batch_size,
            image_width=dataset.w,
            image_height=dataset.h,
            principle_point_x=dataset.cx,
            principle_point_y=dataset.cy,
            focal_length_x=dataset.fl_x,
            focal_length_y=dataset.fl_y,
            ray_near=ray_near,
            ray_far=ray_far,
            transform_matrices=dataset.transform_matrices,
            images=dataset.images,
            state=state, 
            num_ray_samples=num_ray_samples, 
            rng=rng
        )
        print('Loss:', loss)
        if step % 50000 == 0 and step > 0:
            turntable_render(
                30, 64, 32, 32, 4.0, ray_near, ray_far, state, dataset, 
                f'step_{step}_vanilla_turntable_render'
            ) 
    return state

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
    principle_point_x:float,
    principle_point_y:float,
    focal_length_x:float,
    focal_length_y:float,
    ray_near:float,
    ray_far:float,
    transform_matrices:jnp.ndarray,
    images:jnp.ndarray,
    state:TrainState,
    num_ray_samples:int,
    rng:PRNGKeyArray
):
    ray_sample_key, pixel_sample_key = jax.random.split(rng, num=2)
    source_pixels, indices = sample_pixels(
        num_samples=batch_size, 
        image_width=image_width, 
        image_height=image_height, 
        num_images=images.shape[0], 
        rng=pixel_sample_key, 
        images=images
    )

    image_indices, width_indices, height_indices = indices
    get_ray_samples_vmap = jax.vmap(
        get_ray_samples, in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None)
    )
    rays_samples, directions, z_vals = get_ray_samples_vmap(
        width_indices, 
        height_indices, 
        transform_matrices[image_indices], 
        jax.random.split(ray_sample_key, num=batch_size),
        principle_point_x, 
        principle_point_y, 
        focal_length_x, 
        focal_length_y, 
        ray_near, 
        ray_far, 
        num_ray_samples,
        True
    )
    normalized_directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

    def get_output(params, rays, normalized_directions):
        return state.apply_fn({'params': params}, (rays, normalized_directions))
    get_output_sample_vmap = jax.vmap(get_output, in_axes=(None, 0, None))
    get_output_batch_vmap = jax.vmap(get_output_sample_vmap, in_axes=(None, 0, 0))

    def get_rendered_pixel(densities, colors, z_vals, directions):
        return render_pixel(densities, colors, z_vals, directions)
    get_rendered_pixel_vmap = jax.vmap(get_rendered_pixel, in_axes=0)

    def loss_fn(params):
        densities, colors = get_output_batch_vmap(params, rays_samples, normalized_directions)
        rendered_colors, rendered_alphas = get_rendered_pixel_vmap(
            densities, colors, z_vals, directions
        )
        source_alphas = source_pixels[:, -1:]
        source_colors = source_pixels[:, :3]
        loss = jnp.mean((rendered_colors - source_colors)**2)
        return loss
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state

def render_scene(
    num_ray_samples:int,
    patch_size_x:int,
    patch_size_y:int,
    ray_near:float,
    ray_far:float,
    dataset:Dataset, 
    transform_matrix:jnp.ndarray, 
    state:TrainState,
    file_name:Optional[str]='rendered_image'
):
    principle_point_x = dataset.cx
    principle_point_y = dataset.cy
    focal_length_x = dataset.fl_x 
    focal_length_y = dataset.fl_y

    @jax.jit
    def render_ray(x, y, transform_matrix):
        ray_samples, direction, zvals = get_ray_samples(
            x, 
            y,
            transform_matrix, 
            jax.random.PRNGKey(0),
            principle_point_x, 
            principle_point_y, 
            focal_length_x, 
            focal_length_y, 
            ray_near, 
            ray_far, 
            num_ray_samples,
            False
        )
        normalized_direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)

        def get_output(params, rays, normalized_direction):
            return state.apply_fn({'params': params}, (rays, normalized_direction))
        get_output_sample_vmap = jax.vmap(get_output, in_axes=(None, 0, None))
        densities, colors = get_output_sample_vmap(state.params, ray_samples, normalized_direction)
        rendered_pixel = render_pixel(densities, colors, zvals, direction)
        return rendered_pixel
    
    num_patches_x = dataset.w // patch_size_x
    num_patches_y = dataset.h // patch_size_y
    rendered_image = np.ones((dataset.w, dataset.h, 3))
    render_ray_vmap = jax.vmap(
        jax.vmap(render_ray, in_axes=(None, 0, None)), in_axes=(0, None, None)
    )
    
    for x in range(num_patches_x):
        patch_start_x = patch_size_x * x
        patch_end_x = patch_start_x + patch_size_x
        x_coordinates = jnp.arange(patch_start_x, patch_end_x)
        for y in range(num_patches_y):
            patch_start_y = patch_size_y * y
            patch_end_y = patch_start_y + patch_size_y
            y_coordinates = jnp.arange(patch_start_y, patch_end_y)
            rendered_colors, rendered_alphas = render_ray_vmap(
                x_coordinates, y_coordinates, transform_matrix
            )
            rendered_patch = rendered_colors
            rendered_image[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = rendered_patch

    rendered_image = np.nan_to_num(rendered_image)
    rendered_image = np.clip(rendered_image, 0, 1)
    plt.imsave(os.path.join('data/', file_name + '.png'), rendered_image)

def turntable_render(
    num_frames:int, 
    num_ray_samples:int,
    patch_size_x:int,
    patch_size_y:int,
    camera_distance:float, 
    ray_near:float, 
    ray_far:float, 
    state:TrainState, 
    dataset:Dataset,
    file_name:str='turntable_render'
):
    xy_start_position = jnp.array([0.0, -1.0])
    xy_start_position_angle_2d = 0
    z_start_rotation_angle_3d = 0
    angle_delta = 2 * jnp.pi / num_frames

    x_rotation_angle_3d = jnp.pi / 2
    x_rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(x_rotation_angle_3d), -jnp.sin(x_rotation_angle_3d)],
        [0, jnp.sin(x_rotation_angle_3d), jnp.cos(x_rotation_angle_3d)],
    ])

    for i in range(num_frames):
        xy_position_angle_2d = xy_start_position_angle_2d + i * angle_delta
        z_rotation_angle_3d = z_start_rotation_angle_3d + i * angle_delta

        xy_rotation_matrix_2d = jnp.array([
            [jnp.cos(xy_position_angle_2d), -jnp.sin(xy_position_angle_2d)], 
            [jnp.sin(xy_position_angle_2d), jnp.cos(xy_position_angle_2d)]
        ])
        current_xy_position = xy_rotation_matrix_2d @ xy_start_position
    
        z_rotation_matrix = jnp.array([
            [jnp.cos(z_rotation_angle_3d), -jnp.sin(z_rotation_angle_3d), 0],
            [jnp.sin(z_rotation_angle_3d), jnp.cos(z_rotation_angle_3d), 0],
            [0, 0, 1],
        ])

        rotation_matrix = z_rotation_matrix @ x_rotation_matrix
        translation_matrix = jnp.array([
            [current_xy_position[0]],
            [current_xy_position[1]],
            [0],
        ])
        transform_matrix = jnp.concatenate([rotation_matrix, translation_matrix], axis=-1)
        transform_matrix = process_3x4_transform_matrix(transform_matrix, camera_distance)

        render_scene(
            num_ray_samples=num_ray_samples, 
            patch_size_x=patch_size_x, 
            patch_size_y=patch_size_y, 
            ray_near=ray_near, 
            ray_far=ray_far, 
            dataset=dataset, 
            transform_matrix=transform_matrix, 
            state=state,
            file_name=file_name + f'_frame_{i}'
        )

def process_3x4_transform_matrix(original:jnp.ndarray, scale:float):    
    new = jnp.array([
        [original[0, 0], original[0, 1], original[0, 2], original[0, 3] * scale],
        [original[1, 0], original[1, 1], original[1, 2], original[1, 3] * scale],
        [original[2, 0], original[2, 1], original[2, 2], original[2, 3] * scale],
    ])
    return new

def load_lego_dataset(path:str, downscale_factor:int, translation_scale:float):
    with open(os.path.join(path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        current_image_path = path + frame['file_path'][1:] + '.png'
        image = Image.open(current_image_path)
        image = image.resize(
            (image.width // downscale_factor, image.height // downscale_factor),
            resample=Image.NEAREST
        )
        images.append(jnp.array(image))

    transform_matrices = jnp.array(transform_matrices)[:, :3, :]
    process_transform_matrices_vmap = jax.vmap(process_3x4_transform_matrix, in_axes=(0, None))
    transform_matrices = process_transform_matrices_vmap(transform_matrices, translation_scale)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    dataset = Dataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_x'],
        fl_x=1,
        fl_y=1,
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        transform_matrices=transform_matrices,
        images=images
    )
    dataset.fl_x = dataset.cx / jnp.tan(dataset.horizontal_fov / 2)
    dataset.fl_y = dataset.cy / jnp.tan(dataset.vertical_fov / 2)
    return dataset
def load_tiny_dataset():
    data = jnp.load('data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    dataset = Dataset(
        horizontal_fov=0,
        vertical_fov=0,
        fl_x=focal,
        fl_y=focal,
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        transform_matrices=poses,
        images=images
    )
    return dataset

if __name__ == '__main__':
    print('GPU:', jax.devices('gpu'))

    dataset = load_lego_dataset('data/lego', 2, 1)
    print(dataset.horizontal_fov)
    print(dataset.vertical_fov)
    print(dataset.fl_x)
    print(dataset.fl_y)
    print(dataset.cx)
    print(dataset.cy)
    print(dataset.w)
    print(dataset.h)
    print('Images shape:', dataset.images.shape)

    model = VanillaNerf(
        density_mlp_width=256,
        density_mlp_depth=8,
        color_mlp_width=128,
        color_mlp_depth=1,
        exponential_density_activation=False,
        positional_encoding_dim=10,
        directional_encoding_dim=4
    )
    rng = jax.random.PRNGKey(1)
    state = create_train_state(model, rng, 5e-4)

    ray_near = 2.0
    ray_far = 6.0
    assert ray_near < ray_far, 'Ray near must be less than ray far.'

    state = train_loop(
        batch_size=2048,
        num_ray_samples=128,
        ray_near=ray_near,
        ray_far=ray_far,
        training_steps=300000, 
        state=state, 
        dataset=dataset
    )
    turntable_render(
        30, 128, 32, 32, 4.0, ray_near, ray_far, state, dataset, 'vanilla_turntable_render'
    )
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[2], 
        state=state,
        file_name='vanilla_rendered_image_0'
    )
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[4], 
        state=state,
        file_name='vanilla_rendered_image_1'
    )
    render_scene(
        num_ray_samples=128, 
        patch_size_x=32, 
        patch_size_y=32, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[20], 
        state=state,
        file_name='vanilla_rendered_image_2'
    )