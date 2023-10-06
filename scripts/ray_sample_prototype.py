import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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
    print('Min x scale:', min_x_scale, 'Max x scale:', max_x_scale)

    min_y_scale = (box_min - origin[1]) / direction[1]
    max_y_scale = (box_max - origin[1]) / direction[1]
    min_y_scale, max_y_scale = correct_min_max(min_y_scale, max_y_scale)
    min_y_scale, max_y_scale = blow_up_non_intersections(min_y_scale, max_y_scale, ray_far)
    print('Min y scale:', min_y_scale, 'Max y scale:', max_y_scale)
    
    min_z_scale = (box_min - origin[2]) / direction[2]
    max_z_scale = (box_max - origin[2]) / direction[2]
    min_z_scale, max_z_scale = correct_min_max(min_z_scale, max_z_scale)
    min_z_scale, max_z_scale = blow_up_non_intersections(min_z_scale, max_z_scale, ray_far)
    print('Min z scale:', min_z_scale, 'Max z scale:', max_z_scale)

    # The maximum of the minimum axis scales represents the minimum scale at which all axes 
    # have been intersected once.
    min_scale = jnp.max(jnp.linalg.norm(
        jnp.array([
            direction * min_x_scale, direction * min_y_scale, direction * min_z_scale
        ]), axis=-1
    ))

    # The minimum of the maximum axis scales represents the minimum scale at which a single 
    # axis has been intersected twice.
    max_scale = jnp.min(jnp.linalg.norm(
        jnp.array([
            direction * max_x_scale, direction * max_y_scale, direction * max_z_scale
        ]), axis=-1
    ))

    print('Min scale:', min_scale, 'Max Scale:', max_scale)

    return min_scale, max_scale

def test_intersection():
    box_min = 0
    box_max = 1
    ray_near = 0.1
    ray_far = 3
    num_ray_samples = 10
    direction = jnp.array([-1, -0.01, -0.01])
    #irection = jnp.array([-1, -1, -1])
    direction = direction / jnp.linalg.norm(direction)
    origin = jnp.array([2, 1.01, 1.01])
    
    min_scale, max_scale = ray_intersect(origin, direction, ray_far)
    
    ray_scales = jnp.linspace(ray_near, ray_far, num_ray_samples)
    print('Original ray scales:', ray_scales)
    min_scale_repeated = jnp.full(ray_scales.shape, min_scale)
    max_scale_repeated = jnp.full(ray_scales.shape, max_scale)
    ray_scales = jnp.where(ray_scales < min_scale, min_scale_repeated, ray_scales)
    ray_scales = jnp.where(ray_scales > max_scale, max_scale_repeated, ray_scales)
    print('Modified ray scales:', ray_scales)

    repeated_directions = jnp.repeat(jnp.expand_dims(direction, axis=0), num_ray_samples, axis=0)
    repeated_origins = jnp.repeat(jnp.expand_dims(origin, axis=0), num_ray_samples, axis=0)
    ray_scales = jnp.expand_dims(ray_scales, axis=-1)
    scaled_directions = repeated_directions * ray_scales
    ray_samples = scaled_directions + repeated_origins
    
    print('Final ray scales:', ray_scales)
    print('Repeated directions:', repeated_directions)
    print('Scaled directions:', scaled_directions)
    print('Repeated origins:', repeated_origins)
    print('Ray samples:', ray_samples)

    x_scatter = jnp.ravel(ray_samples[:, 0])
    y_scatter = jnp.ravel(ray_samples[:, 1])
    z_scatter = jnp.ravel(ray_samples[:, 2])

    x_scatter_expanded = np.expand_dims(x_scatter, axis=-1)
    y_scatter_expanded = np.expand_dims(y_scatter, axis=-1)
    z_scatter_expanded = np.expand_dims(z_scatter, axis=-1)

    outside_box = np.repeat(np.expand_dims(np.array([1, 0, 0]), axis=0), x_scatter.shape[0], axis=0)
    inside_box = np.repeat(np.expand_dims(np.array([0, 0, 1]), axis=0), x_scatter.shape[0], axis=0)
    colors = np.where(x_scatter_expanded > 1, outside_box, inside_box)
    colors = np.where(x_scatter_expanded < 0, outside_box, colors)
    colors = np.where(y_scatter_expanded > 1, outside_box, colors)
    colors = np.where(y_scatter_expanded < 0, outside_box, colors)
    colors = np.where(z_scatter_expanded > 1, outside_box, colors)
    colors = np.where(z_scatter_expanded < 0, outside_box, colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x_scatter, y_scatter, z_scatter, c=colors)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Transform")
    plt.show()

test_intersection()