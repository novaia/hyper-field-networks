import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional
import os
import json
from PIL import Image

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
        transform_matrices=jnp.array(transform_matrices),
        images=jnp.array(images, dtype=jnp.float32) / 255.0
    )

    # Isolate translation, rotation, and homogenous components.
    translation_component = dataset.transform_matrices[:, :3, -1]
    rotation_component = dataset.transform_matrices[:, :3, :3]
    homogenous_component = dataset.transform_matrices[:, 3:, :]

    # Scale translations.
    translation_component = translation_component * 0.1
    translation_component = jnp.expand_dims(translation_component, axis=-1)
    
    # Recombine translation, rotation, and homogenous components.
    dataset.transform_matrices = jnp.concatenate([
        jnp.concatenate([rotation_component, translation_component], axis=-1),
        homogenous_component,
    ], axis=1)

    scale = 1
    translation = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    dataset.transform_matrices = jnp.concatenate([
        dataset.transform_matrices[:, :, 0:1] * scale,
        dataset.transform_matrices[:, :, 1:2] * -scale,
        dataset.transform_matrices[:, :, 2:3] * -scale,
        dataset.transform_matrices[:, :, 3:4] * scale + translation,
    ], axis=-1)

    # Swap axes.
    first_rows = dataset.transform_matrices[:, 0:1]
    second_rows = dataset.transform_matrices[:, 1:2]
    third_rows = dataset.transform_matrices[:, 2:3]
    fourth_rows = dataset.transform_matrices[:, 3:4]
    dataset.transform_matrices = jnp.concatenate([
        second_rows, third_rows, first_rows, fourth_rows
    ], axis=1)

    return dataset

def correct_min_max(min_scale, max_scale):
    new_min_scale = jnp.where(min_scale > max_scale, max_scale, min_scale)
    new_max_scale = jnp.where(min_scale > max_scale, min_scale, max_scale)
    return new_min_scale, new_max_scale

def blow_up_non_intersections(min_scale, max_scale, ray_far):
    new_min_scale = jnp.where(min_scale < 0, ray_far, min_scale)
    new_max_scale = jnp.where(max_scale < 0, ray_far, max_scale)
    return new_min_scale, new_max_scale

def ray_intersect(origin, direction, ray_far):
    box_min = 0
    box_max = 1
    
    min_x_scale = (box_min - origin[0]) / direction[0]
    max_x_scale = (box_max - origin[0]) / direction[0]
    min_x_scale, max_x_scale = correct_min_max(min_x_scale, max_x_scale)
    min_x_scale, max_x_scale = blow_up_non_intersections(min_x_scale, max_x_scale, ray_far)

    min_y_scale = (box_min - origin[1]) / direction[1]
    max_y_scale = (box_max - origin[1]) / direction[1]
    min_y_scale, max_y_scale = correct_min_max(min_y_scale, max_y_scale)
    min_y_scale, max_y_scale = blow_up_non_intersections(min_y_scale, max_y_scale, ray_far)
    
    min_z_scale = (box_min - origin[2]) / direction[2]
    max_z_scale = (box_max - origin[2]) / direction[2]
    min_z_scale, max_z_scale = correct_min_max(min_z_scale, max_z_scale)
    min_z_scale, max_z_scale = blow_up_non_intersections(min_z_scale, max_z_scale, ray_far)

    # The maximum of the minimum axis scales represents the minimum scale at which all axes 
    # have been intersected once. Rays with direction components less than this scale have yet
    # to enter the AABB.
    min_scale = jnp.max(jnp.linalg.norm(
        jnp.array([
            direction * min_x_scale, direction * min_y_scale, direction * min_z_scale
        ]), axis=-1
    ))

    # The minimum of the maximum axis scales represents the minimum scale at which a single 
    # axis has been intersected twice. Rays with direction components greater than this scale
    # have exited the AABB.
    max_scale = jnp.min(jnp.linalg.norm(
        jnp.array([
            direction * max_x_scale, direction * max_y_scale, direction * max_z_scale
        ]), axis=-1
    ))

    return min_scale, max_scale

def get_ray_samples(
    uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y, ray_near, ray_far, num_ray_samples
):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    origin = transform_matrix[:3, 3] + direction * ray_near
    normalized_direction = direction / jnp.linalg.norm(direction)
    
    min_scale, max_scale = ray_intersect(origin, normalized_direction, ray_far)
    ray_scales = jnp.linspace(ray_near, ray_far, num_ray_samples)

    min_scale_repeated = jnp.full(ray_scales.shape, min_scale)
    max_scale_repeated = jnp.full(ray_scales.shape, max_scale)
    ray_scales = jnp.where(ray_scales < min_scale, min_scale_repeated, ray_scales)
    ray_scales = jnp.where(ray_scales > max_scale, max_scale_repeated, ray_scales)

    repeated_directions = jnp.repeat(jnp.expand_dims(direction, axis=0), num_ray_samples, axis=0)
    repeated_origins = jnp.repeat(jnp.expand_dims(origin, axis=0), num_ray_samples, axis=0)
    ray_scales = jnp.expand_dims(ray_scales, axis=-1)
    scaled_directions = repeated_directions * ray_scales
    ray_samples = scaled_directions + repeated_origins
    
    return ray_samples, repeated_directions

def get_ray_samples_no_clipping(
    uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y, ray_near, ray_far, num_ray_samples
):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    origin = transform_matrix[:3, 3] + direction * ray_near
    normalized_direction = direction / jnp.linalg.norm(direction)
    
    ray_scales = jnp.linspace(ray_near, ray_far, num_ray_samples)

    repeated_directions = jnp.repeat(
        jnp.expand_dims(normalized_direction, axis=0), num_ray_samples, axis=0
    )
    repeated_origins = jnp.repeat(jnp.expand_dims(origin, axis=0), num_ray_samples, axis=0)
    ray_scales = jnp.expand_dims(ray_scales, axis=-1)
    scaled_directions = repeated_directions * ray_scales
    ray_samples = scaled_directions + repeated_origins
    
    return ray_samples, repeated_directions

def plot_ray_samples(ray_samples, plot_name):
    x_scatter = jnp.ravel(ray_samples[:, :, 0])
    y_scatter = jnp.ravel(ray_samples[:, :, 1])
    z_scatter = jnp.ravel(ray_samples[:, :, 2])

    x_scatter_expanded = jnp.expand_dims(x_scatter, axis=-1)
    y_scatter_expanded = jnp.expand_dims(y_scatter, axis=-1)
    z_scatter_expanded = jnp.expand_dims(z_scatter, axis=-1)

    outside_box = np.repeat(
        np.expand_dims(np.array([1, 0, 0]), axis=0), x_scatter.shape[0], axis=0
    )
    inside_box = np.repeat(
        np.expand_dims(np.array([0, 0, 1]), axis=0), x_scatter.shape[0], axis=0
    )
    colors = np.where(x_scatter_expanded > 1.1, outside_box, inside_box)
    colors = np.where(x_scatter_expanded < -0.1, outside_box, colors)
    colors = np.where(y_scatter_expanded > 1.1, outside_box, colors)
    colors = np.where(y_scatter_expanded < -0.1, outside_box, colors)
    colors = np.where(z_scatter_expanded > 1.1, outside_box, colors)
    colors = np.where(z_scatter_expanded < -0.1, outside_box, colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_scatter, y_scatter, z_scatter, c=colors)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(plot_name)
    plt.show()

def display_ray_samples(dataset):
    ray_near = 0.1
    ray_far = 3.0
    num_ray_samples = 5
    width_indices = jnp.full((200,), 256)
    height_indices = jnp.full((200,), 256)
    image_indices = jnp.arange(200)
    get_ray_samples_vmap = jax.vmap(
        get_ray_samples, 
        in_axes=(0, 0, 0, None, None, None, None, None, None, None)
    )
    ray_samples, _ = get_ray_samples_vmap(
        width_indices,
        height_indices,
        dataset.transform_matrices[image_indices],
        dataset.cx,
        dataset.cy,
        dataset.fl_x,
        dataset.fl_y,
        ray_near,
        ray_far,
        num_ray_samples
    )
    
    get_ray_samples_no_clipping_vmap = jax.vmap(
        get_ray_samples_no_clipping, 
        in_axes=(0, 0, 0, None, None, None, None, None, None, None)
    )
    ray_samples_unclipped, _ = get_ray_samples_no_clipping_vmap(
        width_indices,
        height_indices,
        dataset.transform_matrices[image_indices],
        dataset.cx,
        dataset.cy,
        dataset.fl_x,
        dataset.fl_y,
        ray_near,
        ray_far,
        num_ray_samples
    )

    plot_ray_samples(ray_samples, 'Clipped Ray Samples')
    plot_ray_samples(ray_samples_unclipped, 'Unclipped Ray Samples')

dataset = load_dataset('data/generation_0')
display_ray_samples(dataset)