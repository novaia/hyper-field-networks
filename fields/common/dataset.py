import os
import json
from PIL import Image
from typing import Optional
from dataclasses import dataclass
import jax.numpy as jnp
import jax
from fields.common.matrices import process_3x4_transformation_matrix

@dataclass
class NerfDataset:
    horizontal_fov: float
    vertical_fov: float
    fl_x: Optional[float] = None # Focal length x.
    fl_y: Optional[float] = None # Focal length y.
    k1: Optional[float] = None # First radial distortion parameter.
    k2: Optional[float] = None # Second radial distortion parameter.
    p1: Optional[float] = None # Third radial distortion parameter.
    p2: Optional[float] = None # Fourth radial distortion parameter.
    cx: Optional[float] = None # Principal point x.
    cy: Optional[float] = None # Principal point y.
    w: Optional[int] = None # Image width.
    h: Optional[int] = None # Image height.
    aabb_scale: Optional[int] = None # Scale of scene bounding box.
    transform_matrices: Optional[jnp.ndarray] = None
    images: Optional[jnp.ndarray] = None

def load_nerf_dataset(dataset_path:str, downscale_factor:int):
    with open(os.path.join(dataset_path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    frame_data = transforms['frames']
    first_file_path = frame_data[0]['file_path']
    # Process file paths if they're in the original nerf format.
    if not first_file_path.endswith('.png') and first_file_path.startswith('.'):
        process_file_path = lambda path: path[2:] + '.png'
    else:
        process_file_path = lambda path: path

    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        file_path = process_file_path(frame['file_path'])
        image = Image.open(os.path.join(dataset_path, file_path))
        image = image.resize(
            (image.width // downscale_factor, image.height // downscale_factor),
            resample=Image.NEAREST
        )
        images.append(jnp.array(image))

    transform_matrices = jnp.array(transform_matrices)[:, :3, :]
    mean_translation = jnp.mean(jnp.linalg.norm(transform_matrices[:, :, -1], axis=-1))
    translation_scale = (1 / mean_translation) * 2
    process_transform_matrices_vmap = jax.vmap(process_3x4_transformation_matrix, in_axes=(0, None))
    transform_matrices = process_transform_matrices_vmap(transform_matrices, translation_scale)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    dataset = NerfDataset(
        horizontal_fov=transforms['camera_angle_x'],
        vertical_fov=transforms['camera_angle_x'],
        fl_x=1,
        fl_y=1,
        cx=images.shape[1]/2,
        cy=images.shape[2]/2,
        w=images.shape[1],
        h=images.shape[2],
        aabb_scale=1,
        transform_matrices=transform_matrices,
        images=images
    )
    dataset.fl_x = float(dataset.cx / jnp.tan(dataset.horizontal_fov / 2))
    dataset.fl_y = float(dataset.cy / jnp.tan(dataset.vertical_fov / 2))
    return dataset