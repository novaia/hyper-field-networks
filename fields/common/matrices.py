import jax
import jax.numpy as jnp

def process_3x4_transformation_matrix(original:jax.Array, scale:float) -> jax.Array:
    # Note that the translation component is not shifted.
    new = jnp.array([
        [original[1, 0], -original[1, 1], -original[1, 2], original[1, 3] * scale],
        [original[2, 0], -original[2, 1], -original[2, 2], original[2, 3] * scale],
        [original[0, 0], -original[0, 1], -original[0, 2], original[0, 3] * scale],
    ])
    return new

def get_x_rotation_matrix_3d(angle:float) -> jax.Array:
    x_rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(angle), -jnp.sin(angle)],
        [0, jnp.sin(angle), jnp.cos(angle)],
    ])
    return x_rotation_matrix

def get_y_rotation_matrix_3d(angle:float) -> jax.Array:
    y_rotation_matrix = jnp.array([
        [jnp.cos(angle), 0, jnp.sin(angle)],
        [0, 1, 0],
        [-jnp.sin(angle), 0, jnp.cos(angle)],
    ])
    return y_rotation_matrix

def get_z_rotation_matrix_3d(angle:float) -> jax.Array:
    z_rotation_matrix = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle), 0],
        [jnp.sin(angle), jnp.cos(angle), 0],
        [0, 0, 1],
    ])
    return z_rotation_matrix

def get_rotation_matrix_2d(angle:float) -> jax.Array:
    rotation_matrix = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle)], 
        [jnp.sin(angle), jnp.cos(angle)]
    ])
    return rotation_matrix

def get_z_axis_camera_orbit_matrix(angle:float, orbit_distance:float) -> jax.Array:
    # Rotate a unit vector on the xy plane to get the xy offset.
    xy_offset = jnp.array([0.0, -1.0])
    xy_rotation_matrix_2d = get_rotation_matrix_2d(angle)
    xy_offset = xy_rotation_matrix_2d @ xy_offset
    # Rotate camera on x axis by pi/2 (90 degrees) so that it points forward.
    x_rotation_matrix = get_x_rotation_matrix_3d(jnp.pi/2)
    # Rotate camera on z axis so that it points at orbit center when offset.
    z_rotation_matrix = get_z_rotation_matrix_3d(angle)
    # Combine matrices.
    orbit_matrix = z_rotation_matrix @ x_rotation_matrix
    translation_component = jnp.array([[xy_offset[0]], [xy_offset[1]], [0]])
    orbit_matrix = jnp.concatenate([orbit_matrix, translation_component], axis=-1)
    orbit_matrix = process_3x4_transformation_matrix(orbit_matrix, orbit_distance)
    return orbit_matrix