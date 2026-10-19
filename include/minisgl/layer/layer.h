#ifndef MINISGL_LAYER_LAYER_H
#define MINISGL_LAYER_LAYER_H

#include "minisgl/core/tensor.h"
#include "minisgl/model/config.h"
#include <memory>
#include <vector>

namespace minisgl {

class Embedding {
public:
    Embedding(int num_embeddings, int embedding_dim);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids);

    std::shared_ptr<Tensor> weight;
private:
    int num_embeddings_;
    int embedding_dim_;
};

class RMSNorm {
public:
    RMSNorm(int size, float eps);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    void forward_fused_add(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& residual);

    std::shared_ptr<Tensor> weight;
private:
    int size_;
    float eps_;
};

class Linear {
public:
    Linear(int in_features, int out_features, bool has_bias = false);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
private:
    int in_features_;
    int out_features_;
    bool has_bias_;
};

class SwiGLU {
public:
    SwiGLU() = default;
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& gate, const std::shared_ptr<Tensor>& up);
};

class RotaryEmbedding {
public:
    RotaryEmbedding(int head_dim, int max_position, float base);
    void forward(const std::shared_ptr<Tensor>& positions,
                 std::shared_ptr<Tensor>& query,
                 std::shared_ptr<Tensor>& key);

private:
    int head_dim_;
    int max_position_;
    float base_;
    std::shared_ptr<Tensor> cos_sin_cache_;
};

class Attention {
public:
    Attention(int num_heads, int head_dim);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& q,
                                    const std::shared_ptr<Tensor>& k,
                                    const std::shared_ptr<Tensor>& v);
private:
    int num_heads_;
    int head_dim_;
    float scale_;
};

class MoERouter {
public:
    MoERouter(int hidden_size, int num_experts, int top_k, bool norm_topk);
    std::shared_ptr<Tensor> route(const std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& router_weights);

    std::shared_ptr<Tensor> weight;
private:
    int hidden_size_;
    int num_experts_;
    int top_k_;
    bool norm_topk_;
};

} // namespace minisgl

#endif // MINISGL_LAYER_LAYER_H
