#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void fp32_tokenize_kernel(float* input, uint32_t* output, int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(idx < size) 
    {
        output[idx] = reinterpret_cast<uint32_t&>(input[idx]);
    }
}

int main()
{
    int size = 512;
    size_t input_size_bytes = sizeof(float) * size;
    size_t output_size_bytes = sizeof(uint32_t) * size;
    float* h_input;
    float* d_input;
    uint32_t* d_output;
    h_input = (float*)malloc(input_size_bytes);
    cudaMalloc(&d_input, input_size_bytes);
    cudaMalloc(&d_output, output_size_bytes);

    for(int i = 0; i < size; ++i)
    {
        h_input[i] = sinf(2 * 3.14159 * ((float)i / (float)size));
        printf("%f ", h_input[i]);
    }
    printf("\n");
    
    cudaMemcpy(d_input, h_input, input_size_bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    
    fp32_tokenize_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, size);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if(cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "reinterpretFloatToUint launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}
