#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x) \
    do { \
        cudaError_t result = x; \
        if (result != cudaSuccess) \
            throw std::runtime_error( \
                std::string(FILE_LINE " " #x " failed with error ") \
                + cudaGetErrorString(result)); \
    } while(0)

static constexpr float SQRT3 = 1.732050807568877293527446341505872367f;

namespace ngp_volume_rendering 
{

// Static parameters passed to integrate_rays.
struct integrating_descriptor_t
{
    std::uint32_t n_rays;
    std::uint32_t total_samples;
};

// Static parameters passed to integrate_rays_backward.
struct integrating_backward_descriptor_t 
{
    std::uint32_t n_rays;
    std::uint32_t total_samples;
    // Camera's near distance, samples behind the camera's near plane with non-negligible
    // (> ~exp(-10)) densities will be penalized.
    float near_distance;
};

// Static parameters passed to integrate_rays_inference.
struct integrating_inference_descriptor_t 
{
    std::uint32_t n_total_rays;
    std::uint32_t n_rays;
    // See marching_inference_descriptor_t.
    std::uint32_t march_steps_cap;
};

// Static parameters passed to march_rays.
struct marching_descriptor_t 
{
    std::uint32_t n_rays;
    // number of available slots to write generated samples to
    // (i.e. the length of output samples array).
    std::uint32_t total_samples;
    // the length of a minimal ray marching step is calculated 
    // as delta_t := 2 * sqrt(3) / diagonal_n_steps (Appendix E.1 of the NGP paper).
    std::uint32_t diagonal_n_steps;
    // NGP paper: we maintain a cascade of K multiscale occupancy grids, where K = 1 for all synthetic
    // NeRF scenes (single grid) and K ∈ [1, 5] for larger real-world scenes 
    // (up to 5 grids, depending on scene size).
    std::uint32_t K;
    // Density grid resolution, the NPG paper uses 128 for every cascade.
    std::uint32_t G;
    // The half-length of the longest axis of the scene’s bounding box.  
    // E.g. the `bound` of the bounding box [-1, 1]^3 is 1.
    float bound;
    // Next step size is calculated as:
    // clamp(z_val[i] * stepsize_portion, sqrt3/1024.f, 2 * bound * sqrt3/1024.f)
    // where bound is the half-length of the largest axis of the scene’s bounding box, as mentioned
    // in Appendix E.1 of the NGP paper (the intercept theorem).
    float stepsize_portion;
};

// Static parameters passed to march_rays_inference.
struct marching_inference_descriptor_t
{
    // Total number of rays to march.
    std::uint32_t n_total_rays;
    // Number of rays to march in this iteration.
    std::uint32_t n_rays;
    // Same things as diagonal_n_steps, K, and G in MarchingDescriptor.
    std::uint32_t diagonal_n_steps, K, G;
    // Max steps to march, this is only used for early stopping marching, 
    // the minimal ray marching step is still determined by diagonal_n_steps.
    std::uint32_t march_steps_cap;
    // Same thing as bound in MarchingDescriptor.
    float bound;
    // same thing as stepsize_portion in marching_descriptor_t.
    float stepsize_portion;
};

struct morton_3d_descriptor_t
{
    // Number of entries to process.
    std::uint32_t length;
};

// Static parameters passed to pack_density_into_bits.
struct packbits_descriptor_t
{
    std::uint32_t n_bytes;
};

// Functions to register.
void pack_density_into_bits(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void march_rays(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void march_rays_inference(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void morton_3d(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void morton_3d_invert(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void integrate_rays(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void integrate_rays_backward(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

void integrate_rays_inference(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

#ifdef __CUDACC__
inline __device__ float clampf(float val, float lo, float hi) 
{
    return fminf(fmaxf(val, lo), hi);
}
inline __device__ int clampi(int val, int lo, int hi) 
{
    return min(max(val, lo), hi);
}
inline __host__ __device__ float signf(const float x) 
{
    return copysignf(1.0, x);
}

template<typename T> struct vec3 
{
    T x, y, z;
    T __alignment;
    inline __device__ T L_inf() const 
    {
        return fmaxf(fabsf(this->x), fmaxf(fabsf(this->y), fabsf(this->z)));
    }
    inline __device__ vec3 operator+() const 
    {
        return *this;
    }
    inline __device__ vec3 operator+(vec3 const & rhs) const 
    {
        return {this->x + rhs.x, this->y + rhs.y, this->z + rhs.z};
    }
    inline __device__ vec3 operator+(T const & rhs) const 
    {
        return (*this) + vec3 {rhs, rhs, rhs};
    }
    inline __device__ vec3 operator-() const 
    {
        return {-this->x, -this->y, -this->z,};
    }
    inline __device__ vec3 operator-(vec3 const & rhs) const 
    {
        return (*this) + (-rhs);
    }
    inline __device__ vec3 operator-(T const & rhs) const 
    {
        return (*this) - vec3 {rhs, rhs, rhs};
    }
    inline __device__ vec3 operator*(vec3 const & rhs) const 
    {
        return {this->x * rhs.x, this->y * rhs.y, this->z * rhs.z};
    }
    inline __device__ vec3 operator*(T const & rhs) const 
    {
        return (*this) * vec3 {rhs, rhs, rhs};
    }
    inline __device__ vec3 operator/(vec3 const & rhs) const 
    {
        return {this->x / rhs.x, this->y / rhs.y, this->z / rhs.z};
    }
    inline __device__ vec3 operator/(T const & rhs) const 
    {
        return (*this) / vec3 {rhs, rhs, rhs};
    }
};

inline __device__ vec3<float> sign_vec3f(vec3<float> const & v) 
{
    return {signf(v.x), signf(v.y), signf(v.z)};
}
inline __device__ vec3<float> floor_vec3f(vec3<float> const & v) 
{
    return {floorf(v.x), floorf(v.y), floorf(v.z)};
}

template<typename T> 
inline __device__ vec3<T> operator+(T const & lhs, vec3<T> const & rhs) 
{
    return rhs + lhs;
}
template<typename T>
inline __device__ vec3<T> operator-(T const & lhs, vec3<T> const & rhs) 
{
    return -rhs + lhs;
}
template<typename T>
inline __device__ vec3<T> operator*(T const & lhs, vec3<T> const & rhs) 
{
    return rhs * lhs;
}
template<typename T>
inline __device__ vec3<T> operator/(T const & lhs, vec3<T> const & rhs) 
{
    return vec3<T> {lhs, lhs, lhs} / rhs;
}
using vec3f = vec3<float>;
using vec3u = vec3<std::uint32_t>;
#endif

}  // namespace ngp_volume_rendering
