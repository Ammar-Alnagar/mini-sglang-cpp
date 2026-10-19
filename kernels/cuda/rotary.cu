#include <cuda_runtime.h>

namespace minisgl {
namespace cuda {

__global__ void apply_rope_kernel(float* __restrict__ query, float* __restrict__ key,
                                  const int* __restrict__ positions,
                                  const float* __restrict__ cos_sin_cache, int num_tokens,
                                  int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_elements = num_tokens * num_heads * half_dim;

    if (idx >= total_elements)
        return;

    int h = (idx / half_dim) % num_heads;
    int token_idx = idx / (num_heads * half_dim);
    int d = idx % half_dim;

    int pos = positions[token_idx];
    const float* cs = cos_sin_cache + pos * head_dim;
    float cos_val = cs[d];
    float sin_val = cs[half_dim + d];

    float* q = query + (token_idx * num_heads + h) * head_dim;
    float q0 = q[d];
    float q1 = q[half_dim + d];

    q[d] = q0 * cos_val - q1 * sin_val;
    q[half_dim + d] = q0 * sin_val + q1 * cos_val;

    if (key != nullptr) {
        float* k = key + (token_idx * num_heads + h) * head_dim;
        float k0 = k[d];
        float k1 = k[half_dim + d];

        k[d] = k0 * cos_val - k1 * sin_val;
        k[half_dim + d] = k0 * sin_val + k1 * cos_val;
    }
}

void launch_rope(float* query, float* key, const int* positions, const float* cos_sin_cache,
                 int num_tokens, int num_heads, int head_dim, cudaStream_t stream) {
    int half_dim = head_dim / 2;
    int total = num_tokens * num_heads * half_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_rope_kernel<<<blocks, threads, 0, stream>>>(query, key, positions, cos_sin_cache,
                                                      num_tokens, num_heads, head_dim);
}

} // namespace cuda
} // namespace minisgl
