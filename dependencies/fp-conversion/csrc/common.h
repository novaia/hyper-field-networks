#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace fp_conversion
{

struct tokenization_descriptor_t
{
    std::uint32_t mantissa_bits_to_truncate;
    std::uint32_t n_tokens;
};

void launch_tokenization(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
);

}
