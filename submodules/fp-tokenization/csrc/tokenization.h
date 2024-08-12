#pragma once

#include <cstdint>

namespace fp_tokenization
{

struct tokenization_descriptor_t
{
    std::uint32_t n_elements;
}

void fp32_to_token(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);

} // namespace fp_tokenization
