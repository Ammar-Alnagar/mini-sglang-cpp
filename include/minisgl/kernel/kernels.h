#ifndef MINISGL_KERNEL_KERNELS_H
#define MINISGL_KERNEL_KERNELS_H

#ifdef MINISGL_USE_CUDA
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif

namespace minisgl {
namespace cuda {

void launch_rmsnorm(const float* input, const float* weight, float* output, int num_tokens, int hidden_size, float eps, cudaStream_t stream = nullptr);
void launch_rope(float* query, float* key, const int* positions, const float* cos_sin_cache, int num_tokens, int num_heads, int head_dim, cudaStream_t stream = nullptr);
void launch_index_gather(const float* input, const int* indices, float* output, int hidden_size, int num_indices, cudaStream_t stream = nullptr);
void launch_store_kv_page(const float* k, const float* v, float* k_cache, float* v_cache, const int* page_table, int page_size, int num_heads, int head_dim, int num_tokens, cudaStream_t stream = nullptr);

} // namespace cuda
} // namespace minisgl

#endif // MINISGL_KERNEL_KERNELS_H
