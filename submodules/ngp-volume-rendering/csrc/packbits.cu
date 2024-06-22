#include "volrend.h"
#include "serde.h"

namespace ngp_volume_rendering 
{
namespace 
{

__global__ void pack_bits_kernel(
    // Inputs.
    uint32_t const n_bytes, 
    float const* const __restrict__ density_threshold, 
    float const* const __restrict__ density_grid,
    // Outputs.
    bool* const __restrict__ occupied_mask,
    uint8_t* const __restrict__ occupancy_bitfield
){
    uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n_bytes) { return; }

    uint8_t byte = (uint8_t)0x00;

    #pragma unroll
    for(uint8_t idx = 0; idx < 8; ++idx) 
    {
        bool const predicate = (density_grid[i*8+idx] > density_threshold[i*8+idx]);
        occupied_mask[i*8+idx] = predicate;
        byte |= predicate ? ((uint8_t)0x01 << idx) : (uint8_t)0x00;
    }
    occupancy_bitfield[i] = byte;
}

void pack_bits_launcher(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
){
    // Buffer indexing helper.
    uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // Inputs.
    packbits_descriptor_t const &desc = *deserialize<packbits_descriptor_t>(opaque, opaque_len);
    float const* const __restrict__ density_threshold = static_cast<float*>(next_buffer());
    float const* const __restrict__ density_grid = static_cast<float*>(next_buffer());

    // Outputs.
    bool* const __restrict__ occupied_mask = static_cast<bool*>(next_buffer());
    uint8_t * const __restrict__ occupancy_bitfield = static_cast<uint8_t*>(next_buffer());

    // Launch kernel.
    uint32_t static constexpr block_size = 512;
    uint32_t const num_blocks = (desc.n_bytes + block_size - 1) / block_size;
    pack_bits_kernel<<<num_blocks, num_blocks, 0, stream>>>(
        // Inputs
        desc.n_bytes, density_threshold, density_grid,
        // Outputs.
        occupied_mask, occupancy_bitfield
    );
    // Abort on error.
    CUDA_CHECK_THROW(cudaGetLastError());
}

} // namespace

void pack_density_into_bits(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
){
    pack_bits_launcher(stream, buffers, opaque, opaque_len);
}

} // namespace ngp_volume_rendering
