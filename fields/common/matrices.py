import jax
import jax.numpy as jnp

def degrees_to_radians(angle:float) -> float:
    return (angle / 180.0) * jnp.pi 

def process_3x4_transformation_matrix(original:jax.Array, scale:float) -> jax.Array:
    # Note that the translation component is not shifted.
    
    # This is the old way of processing transformation matrices,
    # it's not compatible with transformation matrices from the OpenGL renderer.
    #new = jnp.array([
    #    [original[1, 0], -original[1, 1], -original[1, 2], original[1, 3] * scale],
    #    [original[2, 0], -original[2, 1], -original[2, 2], original[2, 3] * scale],
    #    [original[0, 0], -original[0, 1], -original[0, 2], original[0, 3] * scale],
    #])
    # This is the new way of processing transformation matrices.
    # Changing the signs here seems to have the largest impact on NeRF quality.
    #new = jnp.array([
    #    [original[0, 0], original[0, 1], -original[0, 2], original[0, 3] * scale],
    #    [original[1, 0], original[1, 1], -original[1, 2], original[1, 3] * scale],
    #    [original[2, 0], original[2, 1], -original[2, 2], original[2, 3] * scale],
    #])
    new = jnp.array([
        [original[0, 0], original[0, 1], original[0, 2], original[0, 3] * scale],
        [original[1, 0], original[1, 1], original[1, 2], original[1, 3] * scale],
        [original[2, 0], original[2, 1], original[2, 2], original[2, 3] * scale],
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
    rot_x_matrix = get_x_rotation_matrix_3d(0.0)
    rot_y_matrix = get_y_rotation_matrix_3d(angle)
    rot_matrix = rot_x_matrix @ rot_y_matrix
    position = jnp.array([0.0, 0.0, -1.0], dtype=jnp.float32) * orbit_distance
    position = rot_matrix @ position
    position = jnp.reshape(position, [3, 1])
    orbit_matrix = jnp.concatenate([rot_matrix, position], axis=-1)
    return orbit_matrix
