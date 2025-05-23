#include <stdexcept>
#include <driver_types.h>
#include <vector_types.h>
#include "volrend.h"
#include "serde.h"

namespace ngp_volume_rendering 
{
namespace 
{

// This is the same function as calc_dt in ngp_pl's implementation.
inline __device__ float calc_ds(
    float ray_t, float stepsize_portion, float bound, 
    float inv_grid_res, std::uint32_t diagonal_n_steps
){
    // From appendix E.1 of the NGP paper (the paper sets stepsize_portion=0 for synthetic scenes
    // and 1/256 for others).
    return clampf(
        ray_t * stepsize_portion,
        2 * (float)SQRT3 * fminf(bound, 1.f) / diagonal_n_steps,
        2 * (float)SQRT3 * bound * inv_grid_res
    );
}

inline __device__ uint32_t mip_from_xyz(vec3f pos, uint32_t K) 
{
    if(K == 1) { return 0; }
    float const max_coord = pos.L_inf();
    int exponent;
    frexpf(max_coord, &exponent);
    return static_cast<std::uint32_t>(clampi(exponent, 0, K-1));
}

// Return the finest cascade of occupancy grids that has cell side-length larger than ds (appendix E.2).
inline __device__ uint32_t mip_from_ds(float ds, uint32_t G, uint32_t K) 
{
    if(K == 1) { return 0; }
    int exponent;
    frexpf(ds * G, &exponent);
    return static_cast<std::uint32_t>(clampi(exponent, 0, K-1));
}

inline __device__ vec3u floor_grid_pos(vec3f const & grid_pos, uint32_t const & G) 
{
    return {
        static_cast<uint32_t>(clampi(floorf(grid_pos.x), 0, G-1)),
        static_cast<uint32_t>(clampi(floorf(grid_pos.y), 0, G-1)),
        static_cast<uint32_t>(clampi(floorf(grid_pos.z), 0, G-1)),
    };
}

inline __device__ uint32_t expand_bits(uint32_t v) 
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __device__ uint32_t __morton_3d(uint32_t x, uint32_t y, uint32_t z) 
{
    uint32_t const xx = expand_bits(x);
    uint32_t const yy = expand_bits(y);
    uint32_t const zz = expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __device__ uint32_t __morton_3d(vec3u const & pos) 
{
    return __morton_3d(pos.x, pos.y, pos.z);
}

inline __device__ uint32_t __morton_3d_invert(uint32_t x) 
{
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}

inline __device__ vec3f get_grid_pos_and_intermediates(
    vec3f const & pos, float const ds, float const bound, uint32_t const K, uint32_t const G,
    // Write intermediate values.
    uint32_t * const cascade_ptr, float * const mip_bound_ptr
){
    // Among the grids covering xyz, the finest one with cell side-length larger 
    // than delta_t is queried.
    *cascade_ptr = max(
        mip_from_xyz(pos, K),
        mip_from_ds(ds, G, K)
    );
    // The bound of this mip is [-mip_bound, mip_bound].
    *mip_bound_ptr = fminf(scalbnf(1.f, *cascade_ptr), bound);
    return .5f * (pos / (*mip_bound_ptr) + 1.f) * G;
}

// REF: 
// https://github.com/ashawkey/torch-ngp/blob/b6e080468925f0bb44827b4f8f0ed08291dcf8a9/raymarching/src/raymarching.cu#L312
__global__ void march_rays_kernel(
    // Static.
    uint32_t const n_rays, uint32_t const total_samples, uint32_t const diagonal_n_steps,
    uint32_t const K, uint32_t const G, float const bound, float const stepsize_portion,
    // Inputs.
    float const* const __restrict__ rays_o, // [n_rays, 3]
    float const* const __restrict__ rays_d, // [n_rays, 3]
    float const* const __restrict__ t_starts, // [n_rays]
    float const* const __restrict__ t_ends, // [n_rays]
    float const* const __restrict__ noises, // [n_rays]
    uint8_t const* const __restrict__ occupancy_bitfield, // [K*G*G*G//8]
    // Accumulator for writing a compact output samples array.
    uint32_t* const __restrict__ next_sample_write_location,
    uint32_t* const __restrict__ number_of_exceeded_samples,
    bool* const __restrict__ ray_is_valid,
    // Outputs.
    uint32_t* const __restrict__ rays_n_samples, // [n_rays]
    uint32_t* const __restrict__ rays_sample_startidx, // [n_rays]
    uint32_t* const __restrict__ idcs, // [total_samples]
    float* const __restrict__ xyzs, // [total_samples, 3]
    float* const __restrict__ dirs, // [total_samples, 3]
    float* const __restrict__ dss, // [total_samples]
    float* const __restrict__ z_vals // [total_samples]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_rays) { return; }
    if(*next_sample_write_location >= total_samples) { return; }

    // Inputs.
    vec3f const o = {
        rays_o[i * 3 + 0],
        rays_o[i * 3 + 1],
        rays_o[i * 3 + 2],
    };
    vec3f const d = {
        rays_d[i * 3 + 0],
        rays_d[i * 3 + 1],
        rays_d[i * 3 + 2],
    };
    float const ray_t_start = t_starts[i];
    float const ray_t_end = t_ends[i];

    if(ray_t_end <= ray_t_start) { return; }

    float const ray_noise = noises[i];

    // Precompute.
    uint32_t const G3 = G*G*G;
    vec3f const inv_d = 1.f / d;
    float const inv_G = 1.f / G;

    // March rays but don't write samples to output.
    // Writing is done in another marching pass below.
    uint32_t ray_n_samples = 0;
    float ray_t = ray_t_start;
    ray_t += calc_ds(ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps) * ray_noise;
    while(ray_n_samples < diagonal_n_steps * bound && ray_t < ray_t_end) 
    {
        vec3f const pos = o + ray_t * d;
        float const ds = calc_ds(ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
        uint32_t cascade;
        float mip_bound;
        vec3f const grid_posf = get_grid_pos_and_intermediates(pos, ds, bound, K, G, &cascade, &mip_bound);
        vec3u const grid_pos = floor_grid_pos(grid_posf, G);
        uint32_t const grid_index = cascade * G3 + __morton_3d(grid_pos);
        // (x>>3)==(int)(x/8), (x&7)==(x%8)
        bool const occupied = occupancy_bitfield[grid_index >> 3] & (1 << (grid_index & 7u));

        float new_ray_t = ray_t + ds;
        if(occupied) 
        {
            ++ray_n_samples;
        } 
        else 
        {
            vec3f const next_grid = floor_vec3f(grid_posf + .5f + .5f * sign_vec3f(d));
            vec3f const delta = ((next_grid * inv_G - .5f) * 2.f * mip_bound - pos) * inv_d;
            // Distance to next voxel.
            float const next_t = ray_t + fmaxf(0.0f, fminf(delta.x, fminf(delta.y, delta.z)));
            // Step until next voxel.
            while(new_ray_t < next_t) 
            {
                new_ray_t += calc_ds(new_ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
            }
        }
        ray_t = new_ray_t;
    }

    // Can safely return here because before launching the kernel we have memset'ed every 
    // output array to zeros.
    if(ray_n_samples == 0) 
    {
        ray_is_valid[i] = true;
        return;
    }

    // Cannot check (*next_sample_write_location) + ray_n_samples >= total_samples, a later
    // atomicAdd call on the next_sample_write_location might flip the result of this
    // conditional (TOCTOU race condition).
    // Record the index of the first generated sample on this ray.
    uint32_t const ray_sample_startidx = atomicAdd(next_sample_write_location, ray_n_samples);
    if(ray_sample_startidx + ray_n_samples > total_samples) 
    {
        atomicAdd(number_of_exceeded_samples, ray_n_samples);
        return;
    }

    // Record how many samples are generated along this ray.
    rays_n_samples[i] = ray_n_samples;
    ray_is_valid[i] = true;
    rays_sample_startidx[i] = ray_sample_startidx;

    // Output arrays.
    uint32_t* const __restrict__ ray_idcs = idcs + ray_sample_startidx;
    float* const __restrict__ ray_xyzs = xyzs + ray_sample_startidx * 3;
    float* const __restrict__ ray_dirs = dirs + ray_sample_startidx * 3;
    float* const __restrict__ ray_dss = dss + ray_sample_startidx;
    float* const __restrict__ ray_z_vals = z_vals + ray_sample_startidx;

    // March rays again, this time write sampled points to output.
    uint32_t steps = 0;
    ray_t = ray_t_start;
    ray_t += calc_ds(ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps) * ray_noise;
    // NOTE: we still need the condition (ray_t < ray_t_end) because if a ray never hits 
    // an occupied grid cell, its steps won't increment, adding this condition avoids infinite loops.
    while(steps < ray_n_samples && ray_t < ray_t_end) 
    {
        vec3f const pos = o + ray_t * d;
        float const ds = calc_ds(ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
        uint32_t cascade;
        float mip_bound;
        vec3f const grid_posf = get_grid_pos_and_intermediates(pos, ds, bound, K, G, &cascade, &mip_bound);
        vec3u const grid_pos = floor_grid_pos(grid_posf, G);
        uint32_t const grid_index = cascade * G3 + __morton_3d(grid_pos);
        // (x>>3)==(int)(x/8), (x&7)==(x%8)
        bool const occupied = occupancy_bitfield[grid_index >> 3] & (1 << (grid_index & 7u));

        float new_ray_t = ray_t + ds;
        if(occupied) 
        {
            ray_idcs[steps] = i;  // This sample point comes from the i-th ray.
            ray_xyzs[steps * 3 + 0] = pos.x;
            ray_xyzs[steps * 3 + 1] = pos.y;
            ray_xyzs[steps * 3 + 2] = pos.z;
            ray_dirs[steps * 3 + 0] = d.x;
            ray_dirs[steps * 3 + 1] = d.y;
            ray_dirs[steps * 3 + 2] = d.z;
            ray_dss[steps] = ds;
            ray_z_vals[steps] = ray_t;
            ++steps;
        } 
        else 
        {
            vec3f const next_grid = floor_vec3f(grid_posf + .5f + .5f * sign_vec3f(d));
            vec3f const delta = ((next_grid * inv_G - .5f) * 2.f * mip_bound - pos) * inv_d;
            // Distance to next voxel.
            float const next_t = ray_t + fmaxf(0.0f, fminf(delta.x, fminf(delta.y, delta.z)));
            // Step until next voxel.
            while(new_ray_t < next_t) 
            {
                new_ray_t += calc_ds(new_ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
            }
        }
        ray_t = new_ray_t;
    }
}


__global__ void march_rays_inference_kernel(
    uint32_t const n_total_rays, uint32_t const n_rays, uint32_t const diagonal_n_steps,
    uint32_t const K, uint32_t const G, uint32_t const march_steps_cap, float const bound,
    float const stepsize_portion,
    float const* const __restrict__ rays_o, // [n_total_rays, 3]
    float const* const __restrict__ rays_d, // [n_total_rays, 3]
    float const* const __restrict__ t_starts, // [n_total_rays]
    float const* const __restrict__ t_ends, // [n_total_rays]
    uint8_t const* const __restrict__ occupancy_bitfield, // [K*G*G*G//8]
    bool const* const __restrict__ terminated, // [n_rays]
    uint32_t const* const __restrict__ indices_in, // [n_rays]
    uint32_t * const __restrict__ next_ray_index, // [1]
    uint32_t * const __restrict__ indices_out, // [n_rays]
    uint32_t * const __restrict__ n_samples, // [n_rays]
    float * const __restrict__ t_starts_out, // [n_rays]
    float * const __restrict__ xyzs, // [n_rays, march_steps_cap, 3]
    float * const __restrict__ dss, // [n_rays, march_steps_cap]
    float * const __restrict__ z_vals // [n_rays, march_steps_cap]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_rays) { return; }
    
    uint32_t const ray_idx = terminated[i] ? atomicAdd(next_ray_index, 1u) : indices_in[i];
    indices_out[i] = ray_idx;
    if(ray_idx >= n_total_rays) { return; }

    vec3f const o = {
        rays_o[ray_idx * 3 + 0],
        rays_o[ray_idx * 3 + 1],
        rays_o[ray_idx * 3 + 2],
    };
    vec3f const d = {
        rays_d[ray_idx * 3 + 0],
        rays_d[ray_idx * 3 + 1],
        rays_d[ray_idx * 3 + 2],
    };
    float const ray_t_start = t_starts[ray_idx];
    float const ray_t_end = t_ends[ray_idx];

    if(ray_t_end < ray_t_start) { return; }

    float* const __restrict__ ray_xyzs = xyzs + i * march_steps_cap * 3;
    float* const __restrict__ ray_dss = dss + i * march_steps_cap;
    float* const __restrict__ ray_z_vals = z_vals + i * march_steps_cap;

    uint32_t const G3 = G*G*G;
    vec3f const inv_d = 1.f / d;
    float const inv_G = 1.f / G;

    uint32_t steps = 0;
    float ray_t = ray_t_start;
    while(steps < march_steps_cap && ray_t < ray_t_end) 
    {
        vec3f const pos = o + ray_t * d;
        float const ds = calc_ds(ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
        uint32_t cascade;
        float mip_bound;
        vec3f const grid_posf = get_grid_pos_and_intermediates(pos, ds, bound, K, G, &cascade, &mip_bound);
        vec3u const grid_pos = floor_grid_pos(grid_posf, G);
        uint32_t const grid_index = cascade * G3 + __morton_3d(grid_pos);
        // (x>>3)==(int)(x/8), (x&7)==(x%8)
        bool const occupied = occupancy_bitfield[grid_index >> 3] & (1 << (grid_index & 7u));
        
        float new_ray_t = ray_t + ds;
        if(occupied) 
        {
            ray_xyzs[steps * 3 + 0] = pos.x;
            ray_xyzs[steps * 3 + 1] = pos.y;
            ray_xyzs[steps * 3 + 2] = pos.z;
            ray_dss[steps] = ds;
            ray_z_vals[steps] = ray_t;
            ++steps;
        } 
        else 
        {
            vec3f const next_grid = floor_vec3f(grid_posf + .5f + .5f * sign_vec3f(d));
            vec3f const delta = ((next_grid * inv_G - .5f) * 2.f * mip_bound - pos) * inv_d;
            // Distance to next voxel.
            float const next_t = ray_t + fmaxf(0.0f, fminf(delta.x, fminf(delta.y, delta.z)));
            // Step until next voxel.
            while(new_ray_t < next_t) 
            {
                new_ray_t += calc_ds(new_ray_t, stepsize_portion, bound, inv_G, diagonal_n_steps);
            }
        }
        ray_t = new_ray_t;
    }
    // Write a sample at the far plane (at the farther scene bound) if the ray did not terminate, do
    // not check grid occupancy at this point as it's probably inaccurate.  This avoids "ringing"
    // artifacts that typically occur in scenes with a large bound (> 1), where the ray marching
    // step size is exponential w.r.t. current distance to ray origin.
    if(ray_t >= ray_t_end) 
    {
        vec3f pos = o + ray_t_end * d;
        float ds = calc_ds(ray_t_end, stepsize_portion, bound, inv_G, diagonal_n_steps);
        uint32_t cascade;
        float mip_bound;
        vec3f const grid_posf = get_grid_pos_and_intermediates(pos, ds, bound, K, G, &cascade, &mip_bound);
        vec3u const grid_pos = floor_grid_pos(grid_posf, G);
        uint32_t const grid_index = cascade * G3 + __morton_3d(grid_pos);
        // (x>>3)==(int)(x/8), (x&7)==(x%8)
        bool const occupied = occupancy_bitfield[grid_index >> 3] & (1 << (grid_index & 7u));
        // Only sample a point there if the grid is not marked (as untrainable or empty).
        if (occupied) 
        {
            // If last written point before reaching scene bound is sampled right behind the scene bound,
            // adjust its ds to avoid integrating over a same segment twice.
            if(steps > 0 && ray_dss[steps-1] + ray_z_vals[steps-1] >= ray_t_end) 
            {
                ray_dss[steps-1] = ray_t_end - ray_z_vals[steps-1];
            }
            if(steps < march_steps_cap) 
            {  
                // If there's another slot for writing the last sample.
                ray_xyzs[steps * 3 + 0] = pos.x;
                ray_xyzs[steps * 3 + 1] = pos.y;
                ray_xyzs[steps * 3 + 2] = pos.z;
                ray_dss[steps] = ds;
                ray_z_vals[steps] = ray_t_end;
                ++steps;
            } 
            else 
            {  
                // Otherwise, defer writing the last sample to the next iteration of ray marching.
                ray_t = ray_t_end;
            }
        }
    }
    n_samples[i] = steps;
    t_starts_out[i] = ray_t;
}

__global__ void morton_3d_kernel(
    uint32_t const length,
    uint32_t const* const __restrict__ xyzs, // [length, 3]
    uint32_t* const __restrict__ idcs // [length]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length) { return; }
    idcs[i] = __morton_3d(xyzs[i*3+0], xyzs[i*3+1], xyzs[i*3+2]);
}

__global__ void morton_3d_invert_kernel(
    uint32_t const length,
    uint32_t const* const __restrict__ idcs, // [length]
    uint32_t* const __restrict__ xyzs // [length, 3]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= length) { return; }

    xyzs[i*3+0] = __morton_3d_invert(idcs[i] >> 0);
    xyzs[i*3+1] = __morton_3d_invert(idcs[i] >> 1);
    xyzs[i*3+2] = __morton_3d_invert(idcs[i] >> 2);
}

void march_rays_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    marching_descriptor_t const &desc = *deserialize<marching_descriptor_t>(opaque, opaque_len);
    uint32_t const n_rays = desc.n_rays;
    uint32_t const total_samples = desc.total_samples;
    uint32_t const diagonal_n_steps = desc.diagonal_n_steps;
    uint32_t const K = desc.K;
    uint32_t const G = desc.G;
    float const bound = desc.bound;
    float const stepsize_portion = desc.stepsize_portion;
    float const* const __restrict__ rays_o = static_cast<float*>(next_buffer()); // [n_rays, 3]
    float const* const __restrict__ rays_d = static_cast<float*>(next_buffer()); // [n_rays, 3]
    float const* const __restrict__ t_starts = static_cast<float*>(next_buffer()); // [n_rays]
    float const* const __restrict__ t_ends = static_cast<float*>(next_buffer()); // [n_rays]
    float const* const __restrict__ noises = static_cast<float*>(next_buffer()); // [n_rays]
    uint8_t const* const __restrict__ occupancy_bitfield = static_cast<uint8_t*>(next_buffer()); // [K*G*G*G//8]

    // Helpers.
    uint32_t* const __restrict__ next_sample_write_location = static_cast<uint32_t*>(next_buffer());
    uint32_t* const __restrict__ number_of_exceeded_samples = static_cast<uint32_t*>(next_buffer());
    bool* const __restrict__ ray_is_valid = static_cast<bool*>(next_buffer());

    // Outputs.
    uint32_t* const __restrict__ rays_n_samples = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    uint32_t* const __restrict__ rays_sample_startidx = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    uint32_t* const __restrict__ idcs = static_cast<uint32_t*>(next_buffer()); // [total_samples]
    float* const __restrict__ xyzs = static_cast<float*>(next_buffer()); // [total_samples, 3]
    float* const __restrict__ dirs = static_cast<float*>(next_buffer()); // [total_samples, 3]
    float* const __restrict__ dss = static_cast<float*>(next_buffer()); // [total_samples]
    float* const __restrict__ z_vals = static_cast<float*>(next_buffer()); // [total_samples]

    // Reset helper counter and outputs to zeros.
    CUDA_CHECK_THROW(cudaMemsetAsync(next_sample_write_location, 0x00, sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(number_of_exceeded_samples, 0x00, sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(ray_is_valid, false, n_rays * sizeof(bool), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_n_samples, 0x00, n_rays * sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_sample_startidx, 0x00, n_rays * sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(idcs, 0x00, total_samples * sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(xyzs, 0x00, total_samples * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dirs, 0x00, total_samples * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dss, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(z_vals, 0x00, total_samples * sizeof(float), stream));

    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (n_rays + block_size - 1) / block_size;
    march_rays_kernel<<<num_blocks, block_size, 0, stream>>>(
        // Static.
        n_rays, total_samples, diagonal_n_steps, K, G, bound, stepsize_portion,
        // Inputs.
        rays_o, rays_d, t_starts, t_ends, noises, occupancy_bitfield, 
        next_sample_write_location, number_of_exceeded_samples, ray_is_valid,
        // Outputs.
        rays_n_samples, rays_sample_startidx, idcs, xyzs, dirs, dss, z_vals
    );
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

void march_rays_inference_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    marching_inference_descriptor_t const &desc = 
        *deserialize<marching_inference_descriptor_t>(opaque, opaque_len);
    uint32_t const n_total_rays = desc.n_total_rays;
    uint32_t const n_rays = desc.n_rays;
    uint32_t const diagonal_n_steps = desc.diagonal_n_steps;
    uint32_t const K = desc.K;
    uint32_t const G = desc.G;
    uint32_t const march_steps_cap = desc.march_steps_cap;
    float const bound = desc.bound;
    float const stepsize_portion = desc.stepsize_portion;
    float const* const __restrict__ rays_o = static_cast<float*>(next_buffer()); // [n_total_rays, 3]
    float const* const __restrict__ rays_d = static_cast<float*>(next_buffer()); // [n_total_rays, 3]
    float const* const __restrict__ t_starts = static_cast<float*>(next_buffer()); // [n_total_rays]
    float const* const __restrict__ t_ends = static_cast<float*>(next_buffer()); // [n_total_rays]
    uint8_t const* const __restrict__ occupancy_bitfield = static_cast<uint8_t*>(next_buffer()); // [K*G*G*G//8]
    uint32_t const* const __restrict__ next_ray_index_in = static_cast<uint32_t*>(next_buffer()); // [1]
    bool const* const __restrict__ terminated = static_cast<bool*>(next_buffer()); // [n_rays]
    uint32_t const* const __restrict__ indices_in = static_cast<uint32_t*>(next_buffer()); // [n_rays]

    // Outputs.
    uint32_t* const __restrict__ next_ray_index = static_cast<std::uint32_t *>(next_buffer()); // [1]
    uint32_t* const __restrict__ indices_out = static_cast<std::uint32_t *>(next_buffer()); // [n_rays]
    uint32_t* const __restrict__ n_samples = static_cast<std::uint32_t *>(next_buffer()); // [n_rays]
    float* const __restrict__ t_starts_out = static_cast<float *>(next_buffer()); // [n_rays]
    float* const __restrict__ xyzs = static_cast<float *>(next_buffer()); // [n_rays, march_steps_cap, 3]
    float* const __restrict__ dss = static_cast<float *>(next_buffer()); // [n_rays, march_steps_cap]
    float* const __restrict__ z_vals = static_cast<float *>(next_buffer()); // [n_rays, march_steps_cap]

    // Copy input counter value to output counter.
    CUDA_CHECK_THROW(cudaMemcpyAsync(
        next_ray_index, next_ray_index_in, sizeof(uint32_t), 
        cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream
    ));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    // Initialize output arrays to zeros.
    CUDA_CHECK_THROW(cudaMemsetAsync(indices_out, 0x00, n_rays * sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(n_samples, 0x00, n_rays * sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(t_starts_out, 0x00, n_rays * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(xyzs, 0x00, n_rays * march_steps_cap * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dss, 0x00, n_rays * march_steps_cap * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(z_vals, 0x00, n_rays * march_steps_cap * sizeof(float), stream));
    
    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (n_rays + block_size - 1) / block_size;
    march_rays_inference_kernel<<<num_blocks, block_size, 0, stream>>>(
        n_total_rays, n_rays, diagonal_n_steps, K, G, march_steps_cap, bound, stepsize_portion,
        rays_o, rays_d, t_starts, t_ends, occupancy_bitfield, terminated, indices_in,
        next_ray_index, indices_out, n_samples, t_starts_out, xyzs, dss, z_vals
    );
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

void morton_3d_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    morton_3d_descriptor_t const &desc = *deserialize<morton_3d_descriptor_t>(opaque, opaque_len);
    uint32_t const * const __restrict__ xyzs = static_cast<uint32_t*>(next_buffer()); // [length, 3]
    // Output.
    uint32_t * const __restrict__ idcs = static_cast<uint32_t*>(next_buffer()); // [length]

    // Kernel launch.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (desc.length + block_size - 1) / block_size;
    morton_3d_kernel<<<num_blocks, block_size, 0, stream>>>(desc.length, xyzs, idcs);
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

void morton_3d_invert_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    morton_3d_descriptor_t const &desc = *deserialize<morton_3d_descriptor_t>(opaque, opaque_len);
    uint32_t const* const __restrict__ idcs = static_cast<uint32_t*>(next_buffer()); // [length]
    // Output.
    uint32_t* const __restrict__ xyzs = static_cast<std::uint32_t*>(next_buffer());
    
    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (desc.length + block_size - 1) / block_size;
    morton_3d_invert_kernel<<<num_blocks, block_size, 0, stream>>>(desc.length, idcs, xyzs);
}

} // namespace

void march_rays(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    march_rays_launcher(stream, buffers, opaque, opaque_len);
}

void march_rays_inference(
    cudaStream_t stream, void** buffers, char const *opaque, std::size_t opaqlne_len
){
    march_rays_inference_launcher(stream, buffers, opaque, opaqlne_len);
}

void morton_3d(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    morton_3d_launcher(stream, buffers, opaque, opaque_len);
}

void morton_3d_invert(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    morton_3d_invert_launcher(stream, buffers, opaque, opaque_len);
}

} // namespace ngp_volume_rendering
