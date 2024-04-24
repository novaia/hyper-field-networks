#include <cuda_fp16.h>
#include <iostream>
#include <serde-helper/serde.h>
#include "common.h"

namespace float_tokenization
{
/*std::uint32_t float_to_token(float input)
{
    union
    {
        float in;
        std::int32_t out;
    } data;
    data.in = input;
    std::bitset<sizeof(float) * CHAR_BIT> bits(data.out);
    const unsigned int mantissa_bits_to_truncate = 10;
    unsigned int token_exponent = 0;
    std::uint32_t token = 0;
    for(unsigned int i = 31; i > mantissa_bits_to_truncate; i--)
    {
        if(bits[i])
        {
            token += (1 << token_exponent);
        }
        token_exponent++;
    }
    return token;
}

float token_to_float(std::uint32_t input)
{
    std::bitset<sizeof(std::uint32_t) * CHAR_BIT> token_bits(input);
    std::bitset<sizeof(float) * CHAR_BIT> float_bits(0.0f);
    unsigned int float_bits_offset = 0;
    for(int i = 31; i > -1; i--)
    {
        float_bits[float_bits_offset++] = token_bits[i];
    }
    float output;
    memcpy(&output, &float_bits, sizeof(float));
    return output;
}*/

__global__ void tokenization_kernel(
    __half* input, uint32_t* output, 
    const uint32_t mantissa_bits_to_truncate, const uint32_t n_tokens
){
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid > 0) { return; }
    for(uint32_t i = 0; i < n_tokens; i++)
    {
        uint16_t bits = __half_as_ushort(input[i]);
        output[i] = bits >> mantissa_bits_to_truncate;
    }
}

__global__ void detokenization_kernel(
    uint32_t* input, __half* output,
    const uint32_t mantissa_bits_to_restore, const uint32_t n_tokens
){
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid > 0) { return; }
    for(uint32_t i = 0; i < n_tokens; i++)
    {
        uint16_t bits = static_cast<uint16_t>(input[i]);
        output[i] = __ushort_as_half(bits << mantissa_bits_to_restore);
    }
}

void launch_tokenization(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
){
    tokenization_descriptor_t const &desc = 
        *deserialize<tokenization_descriptor_t>(opaque, opaque_len);
    __half* input = static_cast<__half*>(buffers[0]);
    uint32_t* output = static_cast<uint32_t*>(buffers[1]);
    
    tokenization_kernel<<<1, 32, 0, stream>>>(
        input, output, 
        desc.mantissa_shift, desc.n_tokens
    );
}

void launch_detokenization(
    cudaStream_t stream, void** buffers, const char* opaque, std::size_t opaque_len
){
    tokenization_descriptor_t const &desc = 
        *deserialize<tokenization_descriptor_t>(opaque, opaque_len);
    uint32_t* input = static_cast<uint32_t*>(buffers[0]);
    __half* output = static_cast<__half*>(buffers[1]);
    
    detokenization_kernel<<<1, 32, 0, stream>>>(
        input, output, 
        desc.mantissa_shift, desc.n_tokens
    );
}

} // namespace float_tokenization
