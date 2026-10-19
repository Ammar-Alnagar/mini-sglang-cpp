#include "minisgl/layer/layer.h"
#include "minisgl/kernel/kernels.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace minisgl {

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    weight = Tensor::create({num_embeddings, embedding_dim}, DataType::Float32, DeviceType::CPU);
}

std::shared_ptr<Tensor> Embedding::forward(const std::shared_ptr<Tensor>& input_ids) {
    int64_t batch_size = input_ids->numel();
    auto out = Tensor::create({batch_size, embedding_dim_}, DataType::Float32, input_ids->device());

    if (input_ids->device() == DeviceType::CPU) {
        const int32_t* ids = input_ids->data_ptr<int32_t>();
        const float* w = weight->data_ptr<float>();
        float* dst = out->data_ptr<float>();

        for (int64_t i = 0; i < batch_size; ++i) {
            int32_t id = ids[i];
            if (id >= 0 && id < num_embeddings_) {
                std::memcpy(dst + i * embedding_dim_, w + id * embedding_dim_, embedding_dim_ * sizeof(float));
            } else {
                std::memset(dst + i * embedding_dim_, 0, embedding_dim_ * sizeof(float));
            }
        }
    } else if (input_ids->device() == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
        cuda::launch_index_gather(weight->data_ptr<float>(), input_ids->data_ptr<int32_t>(),
                                  out->data_ptr<float>(), embedding_dim_, static_cast<int>(batch_size));
#endif
    }

    return out;
}

RMSNorm::RMSNorm(int size, float eps) : size_(size), eps_(eps) {
    weight = Tensor::create({size}, DataType::Float32, DeviceType::CPU);
}

std::shared_ptr<Tensor> RMSNorm::forward(const std::shared_ptr<Tensor>& x) {
    int64_t num_tokens = x->numel() / size_;
    auto out = Tensor::create(x->shape(), x->dtype(), x->device());

    if (x->device() == DeviceType::CPU) {
        const float* in_ptr = x->data_ptr<float>();
        const float* w_ptr = weight->data_ptr<float>();
        float* out_ptr = out->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            float sum_sq = 0.0f;
            const float* tok_in = in_ptr + i * size_;
            float* tok_out = out_ptr + i * size_;

            for (int j = 0; j < size_; ++j) {
                sum_sq += tok_in[j] * tok_in[j];
            }
            float inv_rms = 1.0f / std::sqrt(sum_sq / size_ + eps_);
            for (int j = 0; j < size_; ++j) {
                tok_out[j] = tok_in[j] * inv_rms * w_ptr[j];
            }
        }
    } else if (x->device() == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
        cuda::launch_rmsnorm(x->data_ptr<float>(), weight->data_ptr<float>(),
                             out->data_ptr<float>(), static_cast<int>(num_tokens), size_, eps_);
#endif
    }
    return out;
}

void RMSNorm::forward_fused_add(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& residual) {
    if (!residual) {
        residual = x;
        x = forward(x);
        return;
    }

    int64_t num_tokens = x->numel() / size_;
    if (x->device() == DeviceType::CPU) {
        float* x_ptr = x->data_ptr<float>();
        float* res_ptr = residual->data_ptr<float>();
        const float* w_ptr = weight->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            float* tok_x = x_ptr + i * size_;
            float* tok_res = res_ptr + i * size_;

            float sum_sq = 0.0f;
            for (int j = 0; j < size_; ++j) {
                tok_res[j] += tok_x[j];
                sum_sq += tok_res[j] * tok_res[j];
            }
            float inv_rms = 1.0f / std::sqrt(sum_sq / size_ + eps_);
            for (int j = 0; j < size_; ++j) {
                tok_x[j] = tok_res[j] * inv_rms * w_ptr[j];
            }
        }
    } else if (x->device() == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
        cuda::launch_rmsnorm(x->data_ptr<float>(), weight->data_ptr<float>(),
                             x->data_ptr<float>(), static_cast<int>(num_tokens), size_, eps_);
#endif
    }
}

Linear::Linear(int in_features, int out_features, bool has_bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(has_bias) {
    weight = Tensor::create({out_features, in_features}, DataType::Float32, DeviceType::CPU);
    if (has_bias) {
        bias = Tensor::create({out_features}, DataType::Float32, DeviceType::CPU);
    }
}

std::shared_ptr<Tensor> Linear::forward(const std::shared_ptr<Tensor>& x) {
    int64_t num_tokens = x->numel() / in_features_;
    auto out = Tensor::create({num_tokens, out_features_}, DataType::Float32, x->device());

    if (x->device() == DeviceType::CPU) {
        const float* x_ptr = x->data_ptr<float>();
        const float* w_ptr = weight->data_ptr<float>();
        const float* b_ptr = bias ? bias->data_ptr<float>() : nullptr;
        float* out_ptr = out->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            for (int o = 0; o < out_features_; ++o) {
                float val = b_ptr ? b_ptr[o] : 0.0f;
                for (int in = 0; in < in_features_; ++in) {
                    val += x_ptr[i * in_features_ + in] * w_ptr[o * in_features_ + in];
                }
                out_ptr[i * out_features_ + o] = val;
            }
        }
    }
    return out;
}

std::shared_ptr<Tensor> SwiGLU::forward(const std::shared_ptr<Tensor>& gate, const std::shared_ptr<Tensor>& up) {
    auto out = Tensor::create(gate->shape(), gate->dtype(), gate->device());
    int64_t n = gate->numel();

    if (gate->device() == DeviceType::CPU) {
        const float* g = gate->data_ptr<float>();
        const float* u = up->data_ptr<float>();
        float* o = out->data_ptr<float>();

        for (int64_t i = 0; i < n; ++i) {
            float silu_g = g[i] / (1.0f + std::exp(-g[i]));
            o[i] = silu_g * u[i];
        }
    }
    return out;
}

RotaryEmbedding::RotaryEmbedding(int head_dim, int max_position, float base)
    : head_dim_(head_dim), max_position_(max_position), base_(base) {
    cos_sin_cache_ = Tensor::create({max_position, head_dim}, DataType::Float32, DeviceType::CPU);
    float* cache = cos_sin_cache_->data_ptr<float>();

    for (int p = 0; p < max_position; ++p) {
        for (int i = 0; i < head_dim / 2; ++i) {
            float freq = 1.0f / std::pow(base, (2.0f * i) / head_dim);
            float val = p * freq;
            cache[p * head_dim + i] = std::cos(val);
            cache[p * head_dim + head_dim / 2 + i] = std::sin(val);
        }
    }
}

void RotaryEmbedding::forward(const std::shared_ptr<Tensor>& positions,
                             std::shared_ptr<Tensor>& query,
                             std::shared_ptr<Tensor>& key) {
    int64_t num_q = query->numel() / head_dim_;
    if (positions->device() == DeviceType::CPU) {
        const int32_t* pos_ptr = positions->data_ptr<int32_t>();
        const float* cache_ptr = cos_sin_cache_->data_ptr<float>();
        float* q_ptr = query->data_ptr<float>();

        for (int64_t i = 0; i < num_q; ++i) {
            int32_t pos = pos_ptr[i % positions->numel()];
            const float* c_s = cache_ptr + pos * head_dim_;
            float* q = q_ptr + i * head_dim_;

            for (int h = 0; h < head_dim_ / 2; ++h) {
                float cos_val = c_s[h];
                float sin_val = c_s[head_dim_ / 2 + h];
                float q0 = q[h];
                float q1 = q[head_dim_ / 2 + h];
                q[h] = q0 * cos_val - q1 * sin_val;
                q[head_dim_ / 2 + h] = q0 * sin_val + q1 * cos_val;
            }
        }
    } else if (positions->device() == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
        cuda::launch_rope(query->data_ptr<float>(), key ? key->data_ptr<float>() : nullptr,
                         positions->data_ptr<int32_t>(), cos_sin_cache_->data_ptr<float>(),
                         static_cast<int>(num_q), 1, head_dim_);
#endif
    }
}

// Attention
Attention::Attention(int num_heads, int head_dim)
    : num_heads_(num_heads), head_dim_(head_dim), scale_(1.0f / std::sqrt(static_cast<float>(head_dim))) {}

std::shared_ptr<Tensor> Attention::forward(const std::shared_ptr<Tensor>& q,
                                           const std::shared_ptr<Tensor>& k,
                                           const std::shared_ptr<Tensor>& v) {
    int64_t num_tokens = q->shape()[0];
    auto out = Tensor::create({num_tokens, num_heads_ * head_dim_}, DataType::Float32, q->device());

    if (q->device() == DeviceType::CPU) {
        const float* q_ptr = q->data_ptr<float>();
        const float* k_ptr = k->data_ptr<float>();
        const float* v_ptr = v->data_ptr<float>();
        float* out_ptr = out->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            for (int h = 0; h < num_heads_; ++h) {
                std::vector<float> scores(i + 1);
                float max_score = -1e9f;

                for (int64_t j = 0; j <= i; ++j) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim_; ++d) {
                        float q_val = q_ptr[(i * num_heads_ + h) * head_dim_ + d];
                        float k_val = k_ptr[(j * num_heads_ + h) * head_dim_ + d];
                        dot += q_val * k_val;
                    }
                    scores[j] = dot * scale_;
                    if (scores[j] > max_score) max_score = scores[j];
                }

                float sum_exp = 0.0f;
                for (int64_t j = 0; j <= i; ++j) {
                    scores[j] = std::exp(scores[j] - max_score);
                    sum_exp += scores[j];
                }

                for (int d = 0; d < head_dim_; ++d) {
                    float val = 0.0f;
                    for (int64_t j = 0; j <= i; ++j) {
                        float prob = scores[j] / sum_exp;
                        val += prob * v_ptr[(j * num_heads_ + h) * head_dim_ + d];
                    }
                    out_ptr[(i * num_heads_ + h) * head_dim_ + d] = val;
                }
            }
        }
    }

    return out;
}

// MoERouter
MoERouter::MoERouter(int hidden_size, int num_experts, int top_k, bool norm_topk)
    : hidden_size_(hidden_size), num_experts_(num_experts), top_k_(top_k), norm_topk_(norm_topk) {
    weight = Tensor::create({num_experts, hidden_size}, DataType::Float32, DeviceType::CPU);
}

std::shared_ptr<Tensor> MoERouter::route(const std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& router_weights) {
    int64_t num_tokens = x->numel() / hidden_size_;
    auto selected_experts = Tensor::create({num_tokens, top_k_}, DataType::Int32, x->device());
    router_weights = Tensor::create({num_tokens, top_k_}, DataType::Float32, x->device());

    if (x->device() == DeviceType::CPU) {
        const float* x_ptr = x->data_ptr<float>();
        const float* w_ptr = weight->data_ptr<float>();
        int32_t* exp_ptr = selected_experts->data_ptr<int32_t>();
        float* rw_ptr = router_weights->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            std::vector<std::pair<float, int>> logits(num_experts_);
            for (int e = 0; e < num_experts_; ++e) {
                float logit = 0.0f;
                for (int h = 0; h < hidden_size_; ++h) {
                    logit += x_ptr[i * hidden_size_ + h] * w_ptr[e * hidden_size_ + h];
                }
                logits[e] = {logit, e};
            }
            std::partial_sort(logits.begin(), logits.begin() + top_k_, logits.end(),
                              std::greater<std::pair<float, int>>());

            float sum_exp = 0.0f;
            for (int k = 0; k < top_k_; ++k) {
                exp_ptr[i * top_k_ + k] = logits[k].second;
                float prob = std::exp(logits[k].first);
                rw_ptr[i * top_k_ + k] = prob;
                sum_exp += prob;
            }
            if (norm_topk_) {
                for (int k = 0; k < top_k_; ++k) {
                    rw_ptr[i * top_k_ + k] /= sum_exp;
                }
            }
        }
    }
    return selected_experts;
}

} // namespace minisgl
