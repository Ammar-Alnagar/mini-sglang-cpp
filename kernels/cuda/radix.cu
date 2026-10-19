#include <cuda_runtime.h>

namespace minisgl {
namespace cuda {

__global__ void store_kv_page_kernel(const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      float* __restrict__ k_cache,
                                      float* __restrict__ v_cache,
                                      const int* __restrict__ page_table,
                                      int page_size,
                                      int num_heads,
                                      int head_dim,
                                      int num_tokens) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_stride = num_heads * head_dim;
    int total = num_tokens * token_stride;
    if (idx >= total) return;

    int tok = idx / token_stride;
    int h_d = idx % token_stride;

    int page_idx = page_table[tok / page_size];
    int offset_in_page = tok % page_size;

    int dst_idx = (page_idx * page_size + offset_in_page) * token_stride + h_d;

    k_cache[dst_idx] = k[idx];
    v_cache[dst_idx] = v[idx];
}

void launch_store_kv_page(const float* k, const float* v, float* k_cache, float* v_cache,
                          const int* page_table, int page_size, int num_heads, int head_dim, int num_tokens, cudaStream_t stream) {
    int total = num_tokens * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    store_kv_page_kernel<<<blocks, threads, 0, stream>>>(k, v, k_cache, v_cache, page_table, page_size, num_heads, head_dim, num_tokens);
}

} // namespace cuda
} // namespace minisgl
