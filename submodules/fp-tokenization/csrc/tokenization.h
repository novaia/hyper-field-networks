#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace fp_tokenization
{

struct tokenization_descriptor_t
{
    std::uint32_t n_elements;
};

void fp32_to_token(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
void token_to_fp32(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
uint32_t get_fp32_to_token_vocab_size();

void fp32_to_bitfield16(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
void bitfield16_to_fp32(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);

void fp32_to_u8_token(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
void u8_token_to_fp32(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
uint32_t get_fp32_to_u8_token_vocab_size();

void fp32_to_byte_pair_token(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);
void byte_pair_token_to_fp32(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
);

} // namespace fp_tokenization
