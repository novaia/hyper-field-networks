#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "serde.h"
#include "tokenization.h"

namespace fp_tokenization
{

__global__ void fp32_to_token_kernel(float* input, uint32_t* output, uint32_t size) 
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size) 
    {
        __half inter = __float2half(input[idx]);
        output[idx] = static_cast<uint32_t>(reinterpret_cast<uint16_t&>(inter));
    }
}

__global__ void token_to_fp32_kernel(uint32_t* input, float* output, uint32_t size) 
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size) 
    {
        uint16_t inter = static_cast<uint16_t>(input[idx]);
        output[idx] = __half2float(reinterpret_cast<__half&>(inter));
    }
}

void fp32_to_token(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    tokenization_descriptor_t const &desc =
        *deserialize<tokenization_descriptor_t>(opaque, opaque_len);

    float* input = static_cast<float*>(buffers[0]);
    uint32_t* output = static_cast<uint32_t*>(buffers[1]);
    const uint32_t size = desc.n_elements;
    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    fp32_to_token_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size); 
}

void token_to_fp32(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    tokenization_descriptor_t const &desc =
        *deserialize<tokenization_descriptor_t>(opaque, opaque_len);

    uint32_t* input = static_cast<uint32_t*>(buffers[0]);
    float* output = static_cast<float*>(buffers[1]);
    const uint32_t size = desc.n_elements;
    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    token_to_fp32_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size); 
}

uint32_t get_fp32_to_token_vocab_size()
{
    // Any permutation of 16 bits is a valid token.
    constexpr uint32_t vocab_size = 1ULL << 16;  // 2^16
    return vocab_size;
}

__global__ void fp32_to_bitfield16_kernel(float* input, uint32_t* output, uint32_t size)
{
    constexpr uint32_t bitfield_size = 16;
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size) 
    {
        __half inter = __float2half(input[idx]);
        uint16_t token = reinterpret_cast<uint16_t&>(inter);
        for(int k = 0; k < bitfield_size; ++k)
        {
            uint16_t mask_result = (token & (1 << k)) > 0 ? 1 : 0;
            output[idx*bitfield_size] = (uint32_t)mask_result;
        }
    }
}

void fp32_to_bitfield16(
    cudaStream_t stream, void** buffers, char const* opaque, std::size_t opaque_len
){
    tokenization_descriptor_t const &desc =
        *deserialize<tokenization_descriptor_t>(opaque, opaque_len);

    float* input = static_cast<float*>(buffers[0]);
    uint32_t* output = static_cast<uint32_t*>(buffers[1]);
    const uint32_t size = desc.n_elements;
    const int threads_per_block = 256;
    const int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    fp32_to_bitfield16_kernel<<<blocks_per_grid, threads_per_block>>>(input, output, size); 
}

//#define STANDALONE_PROGRAM
#ifdef STANDALONE_PROGRAM
int main()
{
    int size = 512;
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    /* Tokenization. */
    size_t input_size_bytes = sizeof(float) * size;
    size_t output_size_bytes = sizeof(uint32_t) * size;
    float* h_input;
    uint32_t* h_output;
    float* d_input;
    uint32_t* d_output;
    h_input = (float*)malloc(input_size_bytes);
    h_output = (uint32_t*)malloc(output_size_bytes);
    cudaMalloc(&d_input, input_size_bytes);
    cudaMalloc(&d_output, output_size_bytes);

    printf("Input:\n");
    for(int i = 0; i < size; ++i)
    {
        h_input[i] = 4.0f * sinf(2.0f * 3.14159f * ((float)i / (float)size));
        printf("%f ", h_input[i]);
    }
    printf("\n");
    
    cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);
        
    fp32_to_token_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, size);
    
    cudaError_t cuda_status;
    cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess) 
    {
        fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cuda_status));
    }
    cudaMemcpy(h_output, d_output, output_size_bytes, cudaMemcpyDeviceToHost);
    printf("\nToken Output:\n");
    for(int i = 0; i < size; ++i)
    {
        printf("%u ", h_output[i]);
    }
    printf("\n");
    
    /* Detokenization. */
    float* h_output_fp32;
    float* d_output_fp32;
    h_output_fp32 = (float*)malloc(input_size_bytes);
    cudaMalloc(&d_output_fp32, input_size_bytes);

    token_to_fp32_kernel<<<blocks_per_grid, threads_per_block>>>(d_output, d_output_fp32, size);
    
    cuda_status = cudaGetLastError();
    if(cuda_status != cudaSuccess) 
    {
        fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cuda_status));
    }
    cudaMemcpy(h_output_fp32, d_output_fp32, input_size_bytes, cudaMemcpyDeviceToHost);
    printf("\nFloat Output:\n");
    for(int i = 0; i < size; ++i)
    {
        printf("%f ", h_output_fp32[i]);
    }
    printf("\n");
}
#endif // STANDALONE_PROGRAM

} // namespace fp_tokenization
