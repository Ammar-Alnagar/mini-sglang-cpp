#include <cuda_runtime.h>

namespace minisgl {
namespace cuda {

__global__ void index_gather_kernel(const float* __restrict__ input,
                                     const int* __restrict__ indices,
                                     float* __restrict__ output,
                                     int hidden_size,
                                     int num_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_indices * hidden_size;
    if (idx >= total) return;

    int i = idx / hidden_size;
    int h = idx % hidden_size;
    int src_row = indices[i];

    output[idx] = input[src_row * hidden_size + h];
}

void launch_index_gather(const float* input, const int* indices, float* output, int hidden_size, int num_indices, cudaStream_t stream) {
    int total = num_indices * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    index_gather_kernel<<<blocks, threads, 0, stream>>>(input, indices, output, hidden_size, num_indices);
}

} // namespace cuda
} // namespace minisgl
