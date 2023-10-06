import jax
import jax.numpy as jnp

def get_sample_mask(samples):
    box_min = 0
    box_max = 1
    sample_mask_zeros = jnp.zeros((samples.shape[0], 1))
    sample_mask_ones = jnp.ones((samples.shape[0], 1))
    samples_x = samples[:, 0:1]
    samples_y = samples[:, 1:2]
    samples_z = samples[:, 2:3]
    sample_mask = jnp.where(samples_x < box_min, sample_mask_zeros, sample_mask_ones)
    sample_mask = jnp.where(samples_x > box_max, sample_mask_zeros, sample_mask)
    sample_mask = jnp.where(samples_y < box_min, sample_mask_zeros, sample_mask)
    sample_mask = jnp.where(samples_y > box_max, sample_mask_zeros, sample_mask)
    sample_mask = jnp.where(samples_z < box_min, sample_mask_zeros, sample_mask)
    sample_mask = jnp.where(samples_z > box_max, sample_mask_zeros, sample_mask)
    return sample_mask

def get_ray_samples_masked(
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

    ray_samples = repeated_directions * ray_scales + repeated_origins
    sample_mask = get_sample_mask(ray_samples)
    ray_samples = ray_samples * sample_mask
    repeated_directions = repeated_directions * sample_mask
    return ray_samples, repeated_directions