#include "minisgl/model/qwen3.h"

namespace minisgl {

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config, int layer_id)
    : layer_id_(layer_id), config_(config),
      input_layernorm(config.hidden_size, config.rms_norm_eps),
      qkv_proj(config.hidden_size,
               (config.num_qo_heads + 2 * config.num_kv_heads) * config.head_dim),
      o_proj(config.num_qo_heads * config.head_dim, config.hidden_size),
      post_attention_layernorm(config.hidden_size, config.rms_norm_eps),
      gate_up_proj(config.hidden_size, 2 * config.intermediate_size),
      down_proj(config.intermediate_size, config.hidden_size),
      rotary_emb(config.head_dim, config.rotary_config.max_position, config.rotary_config.base) {}

std::shared_ptr<Tensor> Qwen3DecoderLayer::forward(std::shared_ptr<Tensor>& x,
                                                   std::shared_ptr<Tensor>& residual,
                                                   const std::shared_ptr<Tensor>& positions) {
    input_layernorm.forward_fused_add(x, residual);

    // QKV Projection
    auto qkv = qkv_proj.forward(x);
    int64_t num_tokens = qkv->shape()[0];
    int q_dim = config_.num_qo_heads * config_.head_dim;
    int k_dim = config_.num_kv_heads * config_.head_dim;
    int v_dim = config_.num_kv_heads * config_.head_dim;

    auto q = Tensor::create({num_tokens, q_dim}, DataType::Float32, qkv->device());
    auto k = Tensor::create({num_tokens, k_dim}, DataType::Float32, qkv->device());
    auto v = Tensor::create({num_tokens, v_dim}, DataType::Float32, qkv->device());

    if (qkv->device() == DeviceType::CPU) {
        const float* src = qkv->data_ptr<float>();
        float* q_dst = q->data_ptr<float>();
        float* k_dst = k->data_ptr<float>();
        float* v_dst = v->data_ptr<float>();

        int total_dim = q_dim + k_dim + v_dim;
        for (int64_t i = 0; i < num_tokens; ++i) {
            std::memcpy(q_dst + i * q_dim, src + i * total_dim, q_dim * sizeof(float));
            std::memcpy(k_dst + i * k_dim, src + i * total_dim + q_dim, k_dim * sizeof(float));
            std::memcpy(v_dst + i * v_dim, src + i * total_dim + q_dim + k_dim,
                        v_dim * sizeof(float));
        }
    }

    // Apply Rotary Embeddings on Q and K
    rotary_emb.forward(positions, q, k);

    // Compute Scaled Dot Product Attention
    Attention attn(config_.num_qo_heads, config_.head_dim);
    auto attn_val = attn.forward(q, k, v);

    // O Projection
    auto attn_out = o_proj.forward(attn_val);

    post_attention_layernorm.forward_fused_add(attn_out, residual);

    // MLP: Gate + Up, SwiGLU, Down
    auto gate_up = gate_up_proj.forward(attn_out);

    int inter_size = config_.intermediate_size;
    auto gate = Tensor::create({num_tokens, inter_size}, DataType::Float32, gate_up->device());
    auto up = Tensor::create({num_tokens, inter_size}, DataType::Float32, gate_up->device());

    if (gate_up->device() == DeviceType::CPU) {
        const float* gu_ptr = gate_up->data_ptr<float>();
        float* g_ptr = gate->data_ptr<float>();
        float* u_ptr = up->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            for (int j = 0; j < inter_size; ++j) {
                g_ptr[i * inter_size + j] = gu_ptr[i * 2 * inter_size + j];
                u_ptr[i * inter_size + j] = gu_ptr[i * 2 * inter_size + inter_size + j];
            }
        }
    }

    auto mlp_act = swiglu.forward(gate, up);
    auto mlp_out = down_proj.forward(mlp_act);

    return mlp_out;
}

Qwen3ForCausalLM::Qwen3ForCausalLM(const ModelConfig& config)
    : config_(config), embed_tokens(config.vocab_size, config.hidden_size),
      norm(config.hidden_size, config.rms_norm_eps),
      lm_head(config.hidden_size, config.vocab_size) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<Qwen3DecoderLayer>(config, i));
    }
}

void Qwen3ForCausalLM::load_weights(WeightLoader& loader) {
    if (loader.has_tensor("model.embed_tokens.weight")) {
        embed_tokens.weight = loader.load_tensor("model.embed_tokens.weight");
    }
    if (loader.has_tensor("model.norm.weight")) {
        norm.weight = loader.load_tensor("model.norm.weight");
    }
    if (loader.has_tensor("lm_head.weight")) {
        lm_head.weight = loader.load_tensor("lm_head.weight");
    }

    for (int i = 0; i < config_.num_layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        if (loader.has_tensor(p + "input_layernorm.weight")) {
            layers[i]->input_layernorm.weight = loader.load_tensor(p + "input_layernorm.weight");
        }
        if (loader.has_tensor(p + "post_attention_layernorm.weight")) {
            layers[i]->post_attention_layernorm.weight =
                loader.load_tensor(p + "post_attention_layernorm.weight");
        }

        // QKV Proj (merged or separate HF weights)
        if (loader.has_tensor(p + "self_attn.qkv_proj.weight")) {
            layers[i]->qkv_proj.weight = loader.load_tensor(p + "self_attn.qkv_proj.weight");
        } else if (loader.has_tensor(p + "self_attn.q_proj.weight") &&
                   loader.has_tensor(p + "self_attn.k_proj.weight") &&
                   loader.has_tensor(p + "self_attn.v_proj.weight")) {
            auto q_w = loader.load_tensor(p + "self_attn.q_proj.weight");
            auto k_w = loader.load_tensor(p + "self_attn.k_proj.weight");
            auto v_w = loader.load_tensor(p + "self_attn.v_proj.weight");

            int64_t q_size = q_w->size_in_bytes();
            int64_t k_size = k_w->size_in_bytes();
            int64_t v_size = v_w->size_in_bytes();

            auto qkv_w = Tensor::create(
                {q_w->shape()[0] + k_w->shape()[0] + v_w->shape()[0], config_.hidden_size},
                DataType::Float32, DeviceType::CPU);
            char* dst = static_cast<char*>(qkv_w->data());
            std::memcpy(dst, q_w->data(), q_size);
            std::memcpy(dst + q_size, k_w->data(), k_size);
            std::memcpy(dst + q_size + k_size, v_w->data(), v_size);
            layers[i]->qkv_proj.weight = qkv_w;
        }

        if (loader.has_tensor(p + "self_attn.o_proj.weight")) {
            layers[i]->o_proj.weight = loader.load_tensor(p + "self_attn.o_proj.weight");
        }

        // Gate / Up Proj (merged or separate HF weights)
        if (loader.has_tensor(p + "mlp.gate_up_proj.weight")) {
            layers[i]->gate_up_proj.weight = loader.load_tensor(p + "mlp.gate_up_proj.weight");
        } else if (loader.has_tensor(p + "mlp.gate_proj.weight") &&
                   loader.has_tensor(p + "mlp.up_proj.weight")) {
            auto g_w = loader.load_tensor(p + "mlp.gate_proj.weight");
            auto u_w = loader.load_tensor(p + "mlp.up_proj.weight");

            int64_t g_size = g_w->size_in_bytes();
            int64_t u_size = u_w->size_in_bytes();

            auto gu_w = Tensor::create({g_w->shape()[0] + u_w->shape()[0], config_.hidden_size},
                                       DataType::Float32, DeviceType::CPU);
            char* dst = static_cast<char*>(gu_w->data());
            std::memcpy(dst, g_w->data(), g_size);
            std::memcpy(dst + g_size, u_w->data(), u_size);
            layers[i]->gate_up_proj.weight = gu_w;
        }

        if (loader.has_tensor(p + "mlp.down_proj.weight")) {
            layers[i]->down_proj.weight = loader.load_tensor(p + "mlp.down_proj.weight");
        }
    }
}

std::shared_ptr<Tensor> Qwen3ForCausalLM::forward(const std::shared_ptr<Tensor>& input_ids,
                                                  const std::shared_ptr<Tensor>& positions) {
    auto x = embed_tokens.forward(input_ids);
    std::shared_ptr<Tensor> residual = nullptr;

    for (auto& layer : layers) {
        x = layer->forward(x, residual, positions);
    }

    norm.forward_fused_add(x, residual);
    auto logits = lm_head.forward(x);
    return logits;
}

} // namespace minisgl
