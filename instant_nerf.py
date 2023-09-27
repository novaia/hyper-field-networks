import flax.linen as nn
from flax.training import train_state
import jax.numpy as jnp
import jax
import optax
import json
import os
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass

# This is an implementation of the NeRF from the paper:
# "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"

# Calculates the fourth order spherical harmonic encoding for the given directions.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
def fourth_order_sh_encoding(directions):
    num_components = 16
    # Using np temporarily for the mutable array.
    # Need to figure out how to do the component calculation with jnp immutable array.
    components = np.zeros((directions.shape[-1], num_components))

    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    xx = x**2
    yy = y**2
    zz = z**2

    # This could probably be optimized by calculating the components in a 1x16 array, 
    # then copying it three times to get a 3x16 array. This way, all three axes wouldn't
    # have to be indexed every time.
    components[..., 0] = 0.28209479177387814
    components[..., 1] = 0.4886025119029199 * y
    components[..., 2] = 0.4886025119029199 * z
    components[..., 3] = 0.4886025119029199 * x
    components[..., 4] = 1.0925484305920792 * x * y
    components[..., 5] = 1.0925484305920792 * y * z
    components[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
    components[..., 7] = 1.0925484305920792 * x * z
    components[..., 8] = 0.5462742152960396 * (xx - yy)
    components[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
    components[..., 10] = 2.890611442640554 * x * y * z
    components[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
    components[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
    components[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
    components[..., 14] = 1.445305721320277 * z * (xx - yy)
    components[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    return components

def render_volume(positions, directions, deltas, state):
    assertion_text = 'Positions, directions, and deltas must have the same shape.'
    assert positions.shape == directions.shape == deltas.shape, assertion_text
    densities, colors = state.apply_fn((positions, directions))
    
    triangular_mask = jnp.tril(jnp.ones(densities.shape))
    repeated_densities = jnp.repeat(densities, densities.shape[0], axis=0)
    triangular_densities = repeated_densities * triangular_mask
    repeated_deltas = jnp.repeat(deltas, deltas.shape[0], axis=0)
    triangular_deltas = repeated_deltas * triangular_mask

    T_sum = jnp.exp(-jnp.sum(triangular_densities * triangular_deltas))
    rendered_color = jnp.sum(T_sum * (1 - jnp.exp(-densities * deltas)) * colors)
    return rendered_color

class MultiResolutionHashEncoding(nn.Module):
    table_init_key: jax.random.PRNGKey
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

    def hash_function(self, x, table_size, hash_offset):
        pre_xor = x * jnp.array([1, 2654435761, 805459861])
        x = jnp.bitwise_xor(pre_xor[:, 0], pre_xor[:, 1])
        x = jnp.bitwise_xor(x, pre_xor[:, 2])
        x %= table_size
        x += hash_offset
        return x
    
    def __call__(self, x):
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
    hash_table_init_rng: jax.random.PRNGKey
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

def create_train_state(model, rng, learning_rate):
    x = (jnp.ones([1, 3]) / 3, jnp.ones([1, 3]) / 3)
    variables = model.init(rng, x)
    params = variables['params']
    tx = optax.adam(learning_rate)
    ts = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    return ts

def sample_pixels(num_samples, image_width, image_height, num_images, rng, images):
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

def train_loop(batch_size, training_steps, state, dataset):
    for step in range(training_steps):
        rng = jax.random.PRNGKey(step)
        pixels, indices = sample_pixels(
            num_samples=batch_size, 
            image_width=dataset.w, 
            image_height=dataset.h, 
            num_images=dataset.images.shape[0], 
            rng=rng, 
            images=dataset.images
        )

        image_indices, width_indices, height_indices = indices
        # Scale from real canvas dimensions to virtual canvas dimensions.
        x_components = width_indices * dataset.canvas_width_ratio
        y_components = height_indices * dataset.canvas_height_ratio
        z_components = jnp.repeat(jnp.array([dataset.canvas_plane]), x_components.shape[0])
        w_components = jnp.ones(x_components.shape[0])
        print('x_components shape:', x_components.shape)
        print('y_components shape:', y_components.shape)
        print('z_components shape:', z_components.shape)
        print('w_components shape:', w_components.shape)
        print('transform_matrices shape:', dataset.transform_matrices[image_indices].shape)
        rays = jnp.stack([x_components, y_components, z_components, w_components], axis=-1)
        # Transform rays from camera space to world space.
        transform_matrices = dataset.transform_matrices[image_indices]
        rays = jax.vmap(lambda a, b: a @ b, in_axes=0)(transform_matrices, rays)
        print('rays shape:', rays.shape)
        break

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

def load_dataset(path):
    with open(os.path.join(path, 'transforms.json'), 'r') as f:
        transforms = json.load(f)

    locations = []
    directions = []
    images = []
    transform_matrices = []
    for frame in transforms['frames']:
        # Locations and directions can probably be precomputed for all datasets.
        # I don't know why Instant NGP doesn't do this.
        # A: probably because they don't extract locations and directions, and instead
        # use the transform matrix to move the camera space rays into world space.
        transform_matrix = jnp.array(frame['transform_matrix'])
        transform_matrices.append(transform_matrix)
        rotation_scale_matrix = transform_matrix[:3, :3]
        rotation_matrix = rotation_scale_matrix / jnp.linalg.norm(rotation_scale_matrix)
        directions.append(jnp.sum(rotation_matrix, axis=1))
        locations.append(transform_matrix[:3, 3])
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
        locations=jnp.array(locations),
        directions=jnp.array(directions),
        images=jnp.array(images) 
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
    print('Locations shape:', dataset.locations.shape)
    print('Directions shape:', dataset.directions.shape)
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
        batch_size=1000, 
        training_steps=1000, 
        state=state, 
        dataset=dataset
    )