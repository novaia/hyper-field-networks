from typing import Tuple

import jax
import jax.numpy as jnp

from . import impl

def march_rays(
    # static
    total_samples: int,
    diagonal_n_steps: int,
    K: int,
    G: int,
    bound: float,
    stepsize_portion: float,

    # inputs
    rays_o: jax.Array,
    rays_d: jax.Array,
    t_starts: jax.Array,
    t_ends: jax.Array,
    noises: jax.Array,
    occupancy_bitfield: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Given a pack of rays (`rays_o`, `rays_d`), their intersection time with the scene bounding box
    (`t_starts`, `t_ends`), and an occupancy grid (`occupancy_bitfield`), generate samples along
    each ray.

    Inputs:
        total_samples `int`: ,
        diagonal_n_steps `int`: the length of a minimal ray marching step is calculated internally
                                as:
                                    Δ𝑡 := √3 / diagonal_n_steps;
                                the NGP paper uses diagonal_n_steps=1024 (as described in appendix
                                E.1).
        K `int`: total number of cascades of `occupancy_bitfield`
        G `int`: occupancy grid resolution, the paper uses 128 for every cascade
        bound `float`: the half length of the longest axis of the scene’s bounding box,
                       e.g. the `bound` of the bounding box [-1, 1]^3 is 1
        stepsize_portion: next step size is calculated as t * stepsize_portion, the paper uses 1/256

        rays_o `[n_rays, 3]`: ray origins
        rays_d `[n_rays, 3]`: **unit** vectors representing ray directions
        t_starts `[n_rays]`: time of the ray entering the scene bounding box
        t_ends `[n_rays]`: time of the ray leaving the scene bounding box
        noises `broadcastable to [n_rays]`: noises to perturb the starting point of ray marching
        occupancy_bitfield `[K*(G**3)//8]`: the occupancy grid represented as a bit array, grid
                                            cells are laid out in Morton (z-curve) order, as
                                            described in appendix E.2 of the NGP paper

    Returns:
        measured_batch_size_before_compaction `int`: total number of generated samples of all rays
        ray_is_valid `bool` `[n_rays]`: a mask, where a true value denotes the ray's gradients
                                        should flow, even if there are no samples generated for it
        idcs `[total_samples]`: indices indicating which ray the i-th sample comes from.
        rays_n_samples `[n_rays]`: number of samples of each ray, its sum is `total_samples`
                                   referenced below
        rays_sample_startidx `[n_rays]`: indices of each ray's first sample
        xyzs `[total_samples, 3]`: spatial coordinates of the generated samples, invalid array
                                   locations are masked out with zeros
        dirs `[total_samples, 3]`: spatial coordinates of the generated samples, invalid array
                                   locations are masked out with zeros.
        dss `[total_samples]`: `ds`s of each sample, for a more detailed explanation of this
                               notation, see documentation of function `volrendjax.integrate_rays`,
                               invalid array locations are masked out with zeros.
        z_vals `[total_samples]`: samples' distances to their origins, invalid array
                                  locations are masked out with zeros.
    """
    n_rays, _ = rays_o.shape
    noises = jnp.broadcast_to(noises, (n_rays,))

    next_sample_write_location, number_of_exceeded_samples, ray_is_valid, rays_n_samples, rays_sample_startidx, idcs, xyzs, dirs, dss, z_vals = impl.march_rays_p.bind(
        # arrays
        rays_o,
        rays_d,
        t_starts,
        t_ends,
        noises,
        occupancy_bitfield,

        # static args
        total_samples=total_samples,
        diagonal_n_steps=diagonal_n_steps,
        K=K,
        G=G,
        bound=bound,
        stepsize_portion=stepsize_portion,
    )

    measured_batch_size_before_compaction = next_sample_write_location[0] - number_of_exceeded_samples[0]

    return measured_batch_size_before_compaction, ray_is_valid, rays_n_samples, rays_sample_startidx, idcs, xyzs, dirs, dss, z_vals


def march_rays_inference(
    # static
    diagonal_n_steps: int,
    K: int,
    G: int,
    march_steps_cap: int,
    bound: float,
    stepsize_portion: float,

    # inputs
    rays_o: jax.Array,
    rays_d: jax.Array,
    t_starts: jax.Array,
    t_ends: jax.Array,
    occupancy_bitfield: jax.Array,
    next_ray_index_in: jax.Array,
    terminated: jax.Array,
    indices: jax.Array,
):
    """
    Inputs:
        diagonal_n_steps, K, G, bound, stepsize_portion: see explanations in function `march_rays`
        march_steps_cap `int`: maximum steps to march for each ray in this iteration

        rays_o `float` `[n_total_rays, 3]`: ray origins
        rays_d `float` `[n_total_rays, 3]`: ray directions
        t_starts `float` `n_total_rays`: distance of each ray's starting point to its origin
        t_ends `float` `n_total_rays`: distance of each ray's ending point to its origin
        occupancy_bitfield `uint8` `[K*(G**3)//8]`: the occupancy grid represented as a bit array
        next_ray_index_in `uint32`: helper variable to keep record of the latest ray that got rendered
        terminated `bool` `[n_rays]`: output of `integrate_rays_inference`, a binary mask indicating
                                      each ray's termination status
        indices `[n_rays]`: each ray's location in the global arrays

    Returns:
        next_ray_index `uint32` `[1]`: for use in next iteration
        indices `uint32` `[n_rays]`: for use in the integrate_rays_inference immediately after
        n_samples `uint32` `[n_rays]`: number of generated samples of each ray in question
        t_starts `float` `[n_rays]`: advanced values of `t` for use in next iteration
        xyzs `float` `[n_rays, march_steps_cap, 3]`: each sample's XYZ coordinate
        dss `float` `[n_rays, march_steps_cap]`: `ds` of each sample
        z_vals `float` `[n_rays, march_steps_cap]`: distance of each sample to their ray origins
    """
    next_ray_index, indices, n_samples, t_starts_out, xyzs, dss, z_vals = impl.march_rays_inference_p.bind(
        rays_o,
        rays_d,
        t_starts,
        t_ends,
        occupancy_bitfield,
        next_ray_index_in,
        terminated,
        indices,

        diagonal_n_steps=diagonal_n_steps,
        K=K,
        G=G,
        march_steps_cap=march_steps_cap,
        bound=bound,
        stepsize_portion=stepsize_portion,
    )
    t_starts = t_starts.at[indices].set(t_starts_out)
    return next_ray_index, indices, n_samples, t_starts, xyzs, dss, z_vals
