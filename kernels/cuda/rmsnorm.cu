#include <cmath>
#include <cuda_runtime.h>

namespace minisgl {
namespace cuda {

__global__ void rmsnorm_kernel(const float* __restrict__ input, const float* __restrict__ weight,
                               float* __restrict__ output, int hidden_size, float eps) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    const float* x = input + bid * hidden_size;
    float* y = output + bid * hidden_size;

    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        sum += val * val;
    }

    for (int mask = 16; mask > 0; mask /= 2) {
        sum += __shfl_xor_sync(0xffffffff, sum, mask);
    }

    __shared__ float s_variance;
    if (tid == 0) {
        s_variance = rsqrtf(sum / hidden_size + eps);
    }
    __syncthreads();

    float inv_rms = s_variance;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[i] = x[i] * inv_rms * weight[i];
    }
}

void launch_rmsnorm(const float* input, const float* weight, float* output, int num_tokens,
                    int hidden_size, float eps, cudaStream_t stream) {
    int threads = 256;
    if (hidden_size < 256)
        threads = hidden_size;
    rmsnorm_kernel<<<num_tokens, threads, 0, stream>>>(input, weight, output, hidden_size, eps);
}

} // namespace cuda
} // namespace minisgl
