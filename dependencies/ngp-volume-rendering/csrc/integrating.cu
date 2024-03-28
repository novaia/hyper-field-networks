#include <cstdint>
#include <serde-helper/serde.h>
#include "volrend.h"

namespace ngp_volume_rendering 
{
namespace 
{

static constexpr float T_THRESHOLD = 1e-4f;

// Debugging kernel for inspecting data passed to custom op.
__global__ void copy_left_to_right(uint32_t length, float const *lhs, float * const rhs) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < length; i += stride) 
    {
        rhs[i] = lhs[i];
    }
}

__global__ void integrate_rays_kernel(
    // Static arguments.
    uint32_t const n_rays,
    // Input arrays (7).
    uint32_t const* const __restrict__ rays_sample_startidx, // [n_rays]
    uint32_t const* const __restrict__ rays_n_samples, // [n_rays]
    float const* const __restrict__ bgs, // [n_rays, 3]
    float const* const __restrict__ dss, // [total_samples]
    float const* const __restrict__ z_vals, // [total_samples]
    float const* const __restrict__ drgbs, // [total_samples, 4]
    // Helper.
    uint32_t* const __restrict__ measured_batch_size, // [1]
    // Output arrays (2).
    float* const __restrict__ final_rgbds, // [n_rays, 4]
    float* const __restrict__ final_opacities // [n_rays]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_rays) { return; }

    // Input.
    uint32_t const start_idx = rays_sample_startidx[i];
    uint32_t const n_samples = rays_n_samples[i];

    float const* const __restrict__ ray_bgs = bgs + i * 3; // [3]
    float const* const __restrict__ ray_dss = dss + start_idx; // [n_samples]
    float const* const __restrict__ ray_z_vals = z_vals + start_idx; // [n_samples]
    float const* const __restrict__ ray_drgbs = drgbs + start_idx * 4; // [n_samples, 4]

    // Front-to-back composition, with early stop.
    uint32_t sample_idx = 0;
    float ray_depth = 0.f;
    float ray_transmittance = 1.f;
    float r = 0.f, g = 0.f, b = 0.f;
    for(; ray_transmittance > T_THRESHOLD && sample_idx < n_samples; ++sample_idx) 
    {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const alpha = 1.f - __expf(-ray_drgbs[sample_idx * 4] * delta_t);
        float const weight = ray_transmittance * alpha;

        // Composite colors.
        r += weight * ray_drgbs[sample_idx * 4 + 1];
        g += weight * ray_drgbs[sample_idx * 4 + 2];
        b += weight * ray_drgbs[sample_idx * 4 + 3];
        // Composite depth.
        ray_depth += weight * z_val;
        // Decay transmittance (reflects the probability of the ray not hitting this sample).
        ray_transmittance *= 1.f - alpha;
    }

    // Write to global memory.
    // Stop ray marching and set the remaining contribution to zero as soon 
    // as the transmittance of the ray drops below a threshold.
    float const opacity = 1.f - ray_transmittance;
    final_opacities[i] = opacity;
    if(ray_transmittance <= T_THRESHOLD)
    { 
        float idenom = 1.f / opacity;
        final_rgbds[i*4+0] = r * idenom;
        final_rgbds[i*4+1] = g * idenom;
        final_rgbds[i*4+2] = b * idenom;
        final_rgbds[i*4+3] = ray_depth * idenom;
    } 
    else 
    {
        final_rgbds[i*4+0] = r + ray_transmittance * ray_bgs[0];
        final_rgbds[i*4+1] = g + ray_transmittance * ray_bgs[1];
        final_rgbds[i*4+2] = b + ray_transmittance * ray_bgs[2];
        final_rgbds[i*4+3] = ray_depth;
    }

    __shared__ uint32_t kernel_measured_batch_size;
    if(threadIdx.x == 0) { kernel_measured_batch_size = 0; }
    __syncthreads();
    atomicAdd(&kernel_measured_batch_size, sample_idx);
    __syncthreads();
    if(threadIdx.x == 0) { atomicAdd(measured_batch_size, kernel_measured_batch_size); }
}

__global__ void integrate_rays_backward_kernel(
    // Static arguments.
    uint32_t const n_rays,
    float const near_distance,
    // Input arrays.
    uint32_t const* const __restrict__ rays_sample_startidx, // [n_rays]
    uint32_t const* const __restrict__ rays_n_samples,  // [n_rays]
    // Original inputs.
    float const* const __restrict__ bgs, // [n_rays, 3]
    float const* const __restrict__ dss, // [total_samples]
    float const* const __restrict__ z_vals, // [total_samples]
    float const* const __restrict__ drgbs, // [total_samples, 4]
    // Original outputs.
    float const* const __restrict__ final_rgbds, // [n_rays, 4]
    float const* const __restrict__ final_opacities, // [n_rays]
    // Gradient inputs.
    float const* const __restrict__ dL_dfinal_rgbds, // [n_rays, 4]
    // Note: Background color blending is done inside the integrate_rays kernel, 
    // so there is no need to accept a dL_dfinal_opacities parameter, it would be all zeros anyways.
    // Output arrays.
    float* const __restrict__ dL_dbgs, // [n_rays, 3]
    float* const __restrict__ dL_dz_vals, // [total_samples]
    float* const __restrict__ dL_ddrgbs // [total_samples, 4]
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_rays) { return; }

    // Input.
    uint32_t const start_idx = rays_sample_startidx[i];
    uint32_t const n_samples = rays_n_samples[i];

    // Original inputs.
    float const* const __restrict__ ray_bgs = bgs + i * 3;
    float const* const __restrict__ ray_dss = dss + start_idx; // [n_samples]
    float const* const __restrict__ ray_z_vals = z_vals + start_idx; // [n_samples]
    float const* const __restrict__ ray_drgbs = drgbs + start_idx * 4; // [n_samples, 4]

    // Original outputs.
    float const ray_final_rgbd[4] = {
        final_rgbds[i * 4 + 0],
        final_rgbds[i * 4 + 1],
        final_rgbds[i * 4 + 2],
        final_rgbds[i * 4 + 3],
    };
    float const ray_final_opacity = final_opacities[i];

    bool ray_terminated = ray_final_opacity >= 1.f - T_THRESHOLD;

    // Gradient inputs.
    float const ray_dL_dfinal_rgbd[4] = {
        dL_dfinal_rgbds[i * 4 + 0],
        dL_dfinal_rgbds[i * 4 + 1],
        dL_dfinal_rgbds[i * 4 + 2],
        dL_dfinal_rgbds[i * 4 + 3],
    };

    // Outputs.
    float* const __restrict__ ray_dL_dbgs = dL_dbgs + i * 3; // [3]
    float* const __restrict__ ray_dL_dz_vals = dL_dz_vals + start_idx; // [n_samples]
    float* const __restrict__ ray_dL_ddrgbs = dL_ddrgbs + start_idx * 4; // [n_samples, 4]

    // Front-to-back composition, with early stop.
    float transmittance = 1.f;
    float cur_rgb[3] = {0.f, 0.f, 0.f};
    float cur_depth = 0.f;
    for(uint32_t sample_idx = 0; transmittance > T_THRESHOLD && sample_idx < n_samples; ++sample_idx) 
    {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const density = ray_drgbs[sample_idx * 4];
        float const alpha = 1.f - __expf(-density * delta_t);
        float const weight = transmittance * alpha;

        cur_rgb[0] += weight * ray_drgbs[sample_idx * 4 + 1];
        cur_rgb[1] += weight * ray_drgbs[sample_idx * 4 + 2];
        cur_rgb[2] += weight * ray_drgbs[sample_idx * 4 + 3];
        cur_depth += weight * z_val;

        // Decay transmittance before gradient calculation, as transmittance used in gradient
        // calculation is T_{i+1}.  REF: <https://note.kiui.moe/others/nerf_gradient/>
        transmittance *= 1.f - alpha;

        // Set outputs.
        // z_val gradients.
        ray_dL_dz_vals[sample_idx] = weight * ray_dL_dfinal_rgbd[3];
        // density gradients.
        float sample_dL_ddensity = delta_t * (
            // Gradients from final_rgbs.
            + ray_dL_dfinal_rgbd[0] * (
                transmittance * ray_drgbs[sample_idx * 4 + 1] - (ray_final_rgbd[0] - cur_rgb[0])
                - ray_bgs[0] * (ray_terminated ? 0.f : 1.f - ray_final_opacity)
            )
            + ray_dL_dfinal_rgbd[1] * (
                transmittance * ray_drgbs[sample_idx * 4 + 2] - (ray_final_rgbd[1] - cur_rgb[1])
                - ray_bgs[1] * (ray_terminated ? 0.f : 1.f - ray_final_opacity)
            )
            + ray_dL_dfinal_rgbd[2] * (
                transmittance * ray_drgbs[sample_idx * 4 + 3] - (ray_final_rgbd[2] - cur_rgb[2])
                - ray_bgs[2] * (ray_terminated ? 0.f : 1.f - ray_final_opacity)
            )
            // Gradients from depth.
            + ray_dL_dfinal_rgbd[3] * (transmittance * z_val - (ray_final_rgbd[3] - cur_depth))
        );
        // Gradients from regularizations
        // Penalize samples for being behind the camera's near plane. This loss requires there
        // to be samples behind the camera's near plane, so the ray's starting point should only
        // be clipped above zero, instead of being clipped above the near distance.
        // REF: <https://github.com/NVlabs/instant-ngp/commit/2b825d383e11655f46786bc0a67fd0681bfceb60>
        float sample_dReg_ddensity = (density > 4e-5 && z_val < near_distance ? 1e-4f : 0.0f);

        // Gradient scaling, as proposed in 
        // "Floaters No More: Radiance Field Gradient Scaling for Improved Near-Camera Training", EGSR23.
        float grad_scalar = fminf(z_val * z_val, 1.f);
        // Assign density gradients to output.
        ray_dL_ddrgbs[sample_idx * 4 + 0] = grad_scalar * sample_dL_ddensity + sample_dReg_ddensity;
        // Color gradients.
        ray_dL_ddrgbs[sample_idx * 4 + 1] = weight * ray_dL_dfinal_rgbd[0];
        ray_dL_ddrgbs[sample_idx * 4 + 2] = weight * ray_dL_dfinal_rgbd[1];
        ray_dL_ddrgbs[sample_idx * 4 + 3] = weight * ray_dL_dfinal_rgbd[2];
    }

    if(transmittance > T_THRESHOLD) 
    {  
        // Gradients for background colors.
        ray_dL_dbgs[0] = transmittance * ray_dL_dfinal_rgbd[0];
        ray_dL_dbgs[1] = transmittance * ray_dL_dfinal_rgbd[1];
        ray_dL_dbgs[2] = transmittance * ray_dL_dfinal_rgbd[2];
    }
}

__global__ void integrate_rays_inference_kernel(
    uint32_t const n_total_rays,
    uint32_t const n_rays,
    uint32_t const march_steps_cap,
    float const* const __restrict__ rays_bg, // [n_total_rays, 3]
    float const* const __restrict__ rays_rgbd, // [n_total_rays, 4]
    float const* const __restrict__ rays_T, // [n_total_rays]
    uint32_t const* const __restrict__ n_samples, // [n_rays]
    uint32_t const* const __restrict__ indices, // [n_rays]
    float const* const __restrict__ dss, // [n_rays, march_steps_cap]
    float const* const __restrict__ z_vals, // [n_rays, march_steps_cap]
    float const* const __restrict__ drgbs, // [n_rays, march_steps_cap, 4]
    uint32_t* const __restrict__ terminate_cnt, // []
    bool* const __restrict__ terminated, // [n_rays]
    float* const __restrict__ rays_rgbd_out, // [n_rays, 4]
    float* const __restrict__ rays_T_out // [n_rays]
){
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_rays) { return; }

    uint32_t const ray_n_samples = n_samples[i];
    uint32_t const ray_idx = indices[i];

    if(ray_idx < n_total_rays) 
    {
        float const* const __restrict__ ray_dss = dss + i * march_steps_cap;
        float const* const __restrict__ ray_z_vals = z_vals + i * march_steps_cap;
        float const* const __restrict__ ray_drgbs = drgbs + i * march_steps_cap * 4;

        float ray_T = rays_T[ray_idx];
        float r = rays_rgbd[ray_idx * 4 + 0];
        float g = rays_rgbd[ray_idx * 4 + 1];
        float b = rays_rgbd[ray_idx * 4 + 2];
        float ray_depth = rays_rgbd[ray_idx * 4 + 3];
        for(std::uint32_t sample_idx = 0; ray_T > T_THRESHOLD && sample_idx < ray_n_samples; ++sample_idx) 
        {
            float const ds = ray_dss[sample_idx];
            float const z_val = ray_z_vals[sample_idx];
            float const density = ray_drgbs[sample_idx * 4];
            float const alpha = 1.f - __expf(-density * ds);
            float const weight = ray_T * alpha;
            r += weight * ray_drgbs[sample_idx * 4 + 1];
            g += weight * ray_drgbs[sample_idx * 4 + 2];
            b += weight * ray_drgbs[sample_idx * 4 + 3];
            ray_depth += weight * z_val;
            ray_T *= (1.f - alpha);
        }

        if(ray_T <= T_THRESHOLD) 
        {
            float const denom = 1.f - ray_T;
            float const idenom = 1.f / denom;
            terminated[i] = true;
            rays_T_out[i] = 0.f;
            rays_rgbd_out[i*4+0] = r * idenom;
            rays_rgbd_out[i*4+1] = g * idenom;
            rays_rgbd_out[i*4+2] = b * idenom;
            rays_rgbd_out[i*4+3] = ray_depth * idenom;
        } 
        else 
        {
            terminated[i] = ray_n_samples < march_steps_cap;
            rays_rgbd_out[i*4+3] = ray_depth;
            rays_T_out[i] = ray_T;
            if(terminated[i]) 
            {
                rays_rgbd_out[i*4+0] = r + ray_T * rays_bg[ray_idx*3+0];
                rays_rgbd_out[i*4+1] = g + ray_T * rays_bg[ray_idx*3+1];
                rays_rgbd_out[i*4+2] = b + ray_T * rays_bg[ray_idx*3+2];
            }
            else 
            {
                rays_rgbd_out[i*4+0] = r;
                rays_rgbd_out[i*4+1] = g;
                rays_rgbd_out[i*4+2] = b;
            }
        }
    }

    __shared__ uint32_t kernel_terminate_cnt;
    if(threadIdx.x == 0) { kernel_terminate_cnt = 0; }
    __syncthreads();
    if(terminated[i]) { atomicAdd(&kernel_terminate_cnt, 1u); }
    __syncthreads();
    if(threadIdx.x == 0) { atomicAdd(terminate_cnt, kernel_terminate_cnt); }
}

void integrate_rays_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    integrating_descriptor_t const &desc = *deserialize<integrating_descriptor_t>(opaque, opaque_len);
    uint32_t const n_rays = desc.n_rays;
    uint32_t const total_samples = desc.total_samples;
    uint32_t const* const __restrict__ rays_sample_startidx = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    uint32_t const* const __restrict__ rays_n_samples = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    float const* const __restrict__ bgs = static_cast<float*>(next_buffer()); // [n_rays, 3]
    float const* const __restrict__ dss = static_cast<float*>(next_buffer()); // [total_samples]
    float const* const __restrict__ z_vals = static_cast<float*>(next_buffer()); // [total_samples]
    float const* const __restrict__ drgbs = static_cast<float*>(next_buffer()); // [total_samples, 4]

    // Helper counter for measured_batch_size.
    uint32_t* const __restrict__ measured_batch_size = static_cast<std::uint32_t *>(next_buffer()); // [1]

    // Outpus.
    float* const __restrict__ final_rgbds = static_cast<float*>(next_buffer()); // [n_rays, 4]
    float* const __restrict__ final_opacities = static_cast<float*>(next_buffer()); // [n_rays]

    // Reset all outputs to zero.
    CUDA_CHECK_THROW(cudaMemsetAsync(measured_batch_size, 0x00, sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(final_rgbds, 0x00, n_rays * 4 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(final_opacities, 0x00, n_rays * sizeof(float), stream));

    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (n_rays + block_size - 1) / block_size;
    integrate_rays_kernel<<<num_blocks, block_size, 0, stream>>>(
        // Static arguments.
        n_rays,
        // Input arrays (7).
        rays_sample_startidx, rays_n_samples, bgs, dss, z_vals, drgbs,
        // Helper counter.
        measured_batch_size,
        // Output arrays (2).
        final_rgbds,
        final_opacities
    );
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

void integrate_rays_backward_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    integrating_backward_descriptor_t const &desc =
        *deserialize<integrating_backward_descriptor_t>(opaque, opaque_len);
    uint32_t const n_rays = desc.n_rays;
    uint32_t const total_samples = desc.total_samples;
    float near_distance = desc.near_distance;
    uint32_t const* const __restrict__ rays_sample_startidx = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    uint32_t const* const __restrict__ rays_n_samples = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    // Original inputs.
    float const* const __restrict__ bgs = static_cast<float*>(next_buffer()); // [n_rays, 3]
    float const* const __restrict__ dss = static_cast<float*>(next_buffer()); // [total_samples]
    float const* const __restrict__ z_vals = static_cast<float*>(next_buffer()); // [total_samples]
    float const* const __restrict__ drgbs = static_cast<float*>(next_buffer()); // [total_samples, 4]
    float const* const __restrict__ final_rgbds = static_cast<float*>(next_buffer()); // [n_rays, 4]
    float const* const __restrict__ final_opacities = static_cast<float*>(next_buffer()); // [n_rays]
    // Gradient inputs.
    float const* const __restrict__ dL_dfinal_rgbds = static_cast<float*>(next_buffer()); // [n_rays, 4]

    // Outputs.
    float* const __restrict__ dL_dbgs = static_cast<float*>(next_buffer()); // [n_rays, 3]
    float* const __restrict__ dL_dz_vals = static_cast<float*>(next_buffer()); // [total_samples]
    float* const __restrict__ dL_ddrgbs = static_cast<float*>(next_buffer()); // [total_samples, 4]
    
    // Reset all outputs to zero.
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dbgs, 0x00, n_rays * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dz_vals, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddrgbs, 0x00, total_samples * 4 * sizeof(float), stream));

    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (n_rays + block_size - 1) / block_size;
    integrate_rays_backward_kernel<<<num_blocks, block_size, 0, stream>>>(
        // Static arguments.
        n_rays, near_distance,
        // Input arrays.
        rays_sample_startidx, rays_n_samples,
        // Original inputs.
        bgs, dss, z_vals, drgbs,
        // Original outputs.
        final_rgbds, final_opacities,
        // Gradient inputs.
        dL_dfinal_rgbds,
        // Output arrays.
        dL_dbgs, dL_dz_vals, dL_ddrgbs
    );
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

void integrate_rays_inference_launcher(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len) 
{
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    integrating_inference_descriptor_t const &desc =
        *deserialize<integrating_inference_descriptor_t>(opaque, opaque_len);
    uint32_t const n_total_rays = desc.n_total_rays;
    uint32_t const n_rays = desc.n_rays;
    uint32_t const march_steps_cap = desc.march_steps_cap;
    float const* const __restrict__ rays_bg = static_cast<float*>(next_buffer()); // [n_total_rays, 3]
    float const* const __restrict__ rays_rgbd = static_cast<float*>(next_buffer()); // [n_total_rays, 4]
    float const* const __restrict__ rays_T = static_cast<float*>(next_buffer()); // [n_total_rays]
    uint32_t const* const __restrict__ n_samples = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    uint32_t const* const __restrict__ indices = static_cast<uint32_t*>(next_buffer()); // [n_rays]
    float const* const __restrict__ dss = static_cast<float*>(next_buffer()); // [n_rays, march_steps_cap]
    float const* const __restrict__ z_vals = static_cast<float*>(next_buffer()); // [n_rays, march_steps_cap]
    float const* const __restrict__ drgbs = static_cast<float*>(next_buffer()); // [n_rays, march_steps_cap, 4]

    // Outputs.
    uint32_t* const __restrict__ terminate_cnt = static_cast<uint32_t*>(next_buffer()); // [1]
    bool* const __restrict__ terminated = static_cast<bool*>(next_buffer()); // [n_rays]
    float* const __restrict__ rays_rgbd_out = static_cast<float*>(next_buffer()); // [n_rays, 4]
    float* const __restrict__ rays_T_out = static_cast<float*>(next_buffer()); // [n_rays]

    // Reset all outputs to zero.
    CUDA_CHECK_THROW(cudaMemsetAsync(terminate_cnt, 0x00, sizeof(uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(terminated, false, n_rays * sizeof(bool), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_rgbd_out, 0x00, n_rays * 4 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_T_out, 0x00, n_rays * sizeof(float), stream));

    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (n_rays + block_size - 1) / block_size;
    integrate_rays_inference_kernel<<<num_blocks, block_size, 1 * sizeof(uint32_t), stream>>>(
        n_total_rays, n_rays, march_steps_cap,
        rays_bg, rays_rgbd, rays_T, n_samples, indices, dss, z_vals, drgbs,
        terminate_cnt, terminated, rays_rgbd_out, rays_T_out
    );
}

} // namespace

// Functions to register.
void integrate_rays(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    integrate_rays_launcher(stream, buffers, opaque, opaque_len);
}

void integrate_rays_backward(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    integrate_rays_backward_launcher(stream, buffers, opaque, opaque_len);
}

void integrate_rays_inference(
    cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len
){
    integrate_rays_inference_launcher(stream, buffers, opaque, opaque_len);
}

} // namespace ngp_volume_rendering.
