#include "minisgl/model/qwen3_moe.h"

namespace minisgl {

Qwen3MoEDecoderLayer::Qwen3MoEDecoderLayer(const ModelConfig& config, int layer_id)
    : layer_id_(layer_id),
      config_(config),
      input_layernorm(config.hidden_size, config.rms_norm_eps),
      qkv_proj(config.hidden_size, (config.num_qo_heads + 2 * config.num_kv_heads) * config.head_dim),
      o_proj(config.num_qo_heads * config.head_dim, config.hidden_size),
      post_attention_layernorm(config.hidden_size, config.rms_norm_eps),
      router(config.hidden_size, config.num_experts, config.num_experts_per_tok, config.norm_topk_prob),
      rotary_emb(config.head_dim, config.rotary_config.max_position, config.rotary_config.base) {
    for (int e = 0; e < config.num_experts; ++e) {
        expert_gate_up.push_back(std::make_unique<Linear>(config.hidden_size, 2 * config.moe_intermediate_size));
        expert_down.push_back(std::make_unique<Linear>(config.moe_intermediate_size, config.hidden_size));
    }
}

std::shared_ptr<Tensor> Qwen3MoEDecoderLayer::forward(std::shared_ptr<Tensor>& x,
                                                      std::shared_ptr<Tensor>& residual,
                                                      const std::shared_ptr<Tensor>& positions) {
    input_layernorm.forward_fused_add(x, residual);

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
            std::memcpy(v_dst + i * v_dim, src + i * total_dim + q_dim + k_dim, v_dim * sizeof(float));
        }
    }

    rotary_emb.forward(positions, q, k);

    Attention attn(config_.num_qo_heads, config_.head_dim);
    auto attn_val = attn.forward(q, k, v);

    auto attn_out = o_proj.forward(attn_val);

    post_attention_layernorm.forward_fused_add(attn_out, residual);

    std::shared_ptr<Tensor> router_weights;
    auto selected_experts = router.route(attn_out, router_weights);

    auto moe_out = Tensor::create({num_tokens, config_.hidden_size}, DataType::Float32, attn_out->device());

    if (attn_out->device() == DeviceType::CPU) {
        const int32_t* exp_ptr = selected_experts->data_ptr<int32_t>();
        const float* rw_ptr = router_weights->data_ptr<float>();
        const float* in_ptr = attn_out->data_ptr<float>();
        float* out_ptr = moe_out->data_ptr<float>();

        for (int64_t i = 0; i < num_tokens; ++i) {
            auto tok_tensor = Tensor::create({1, config_.hidden_size}, DataType::Float32, DeviceType::CPU);
            std::memcpy(tok_tensor->data(), in_ptr + i * config_.hidden_size, config_.hidden_size * sizeof(float));

            for (int k_idx = 0; k_idx < config_.num_experts_per_tok; ++k_idx) {
                int exp_id = exp_ptr[i * config_.num_experts_per_tok + k_idx];
                float weight = rw_ptr[i * config_.num_experts_per_tok + k_idx];

                auto gate_up = expert_gate_up[exp_id]->forward(tok_tensor);
                int inter_size = config_.moe_intermediate_size;

                auto gate = Tensor::create({1, inter_size}, DataType::Float32, DeviceType::CPU);
                auto up = Tensor::create({1, inter_size}, DataType::Float32, DeviceType::CPU);

                const float* gu_p = gate_up->data_ptr<float>();
                float* g_p = gate->data_ptr<float>();
                float* u_p = up->data_ptr<float>();

                for (int j = 0; j < inter_size; ++j) {
                    g_p[j] = gu_p[j];
                    u_p[j] = gu_p[inter_size + j];
                }

                auto act = swiglu.forward(gate, up);
                auto exp_res = expert_down[exp_id]->forward(act);

                const float* exp_res_ptr = exp_res->data_ptr<float>();
                float* dst_ptr = out_ptr + i * config_.hidden_size;
                for (int h = 0; h < config_.hidden_size; ++h) {
                    dst_ptr[h] += weight * exp_res_ptr[h];
                }
            }
        }
    }

    return moe_out;
}

Qwen3MoEForCausalLM::Qwen3MoEForCausalLM(const ModelConfig& config)
    : config_(config),
      embed_tokens(config.vocab_size, config.hidden_size),
      norm(config.hidden_size, config.rms_norm_eps),
      lm_head(config.hidden_size, config.vocab_size) {
    for (int i = 0; i < config.num_layers; ++i) {
        layers.push_back(std::make_unique<Qwen3MoEDecoderLayer>(config, i));
    }
}

void Qwen3MoEForCausalLM::load_weights(WeightLoader& loader) {
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
            layers[i]->post_attention_layernorm.weight = loader.load_tensor(p + "post_attention_layernorm.weight");
        }

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

            auto qkv_w = Tensor::create({q_w->shape()[0] + k_w->shape()[0] + v_w->shape()[0], config_.hidden_size}, DataType::Float32, DeviceType::CPU);
            char* dst = static_cast<char*>(qkv_w->data());
            std::memcpy(dst, q_w->data(), q_size);
            std::memcpy(dst + q_size, k_w->data(), k_size);
            std::memcpy(dst + q_size + k_size, v_w->data(), v_size);
            layers[i]->qkv_proj.weight = qkv_w;
        }

        if (loader.has_tensor(p + "self_attn.o_proj.weight")) {
            layers[i]->o_proj.weight = loader.load_tensor(p + "self_attn.o_proj.weight");
        }

        if (loader.has_tensor(p + "mlp.gate.weight")) {
            layers[i]->router.weight = loader.load_tensor(p + "mlp.gate.weight");
        }

        for (int e = 0; e < config_.num_experts; ++e) {
            std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
            if (loader.has_tensor(ep + "gate_up_proj.weight")) {
                layers[i]->expert_gate_up[e]->weight = loader.load_tensor(ep + "gate_up_proj.weight");
            } else if (loader.has_tensor(ep + "gate_proj.weight") && loader.has_tensor(ep + "up_proj.weight")) {
                auto g_w = loader.load_tensor(ep + "gate_proj.weight");
                auto u_w = loader.load_tensor(ep + "up_proj.weight");
                int64_t g_size = g_w->size_in_bytes();
                int64_t u_size = u_w->size_in_bytes();
                auto gu_w = Tensor::create({g_w->shape()[0] + u_w->shape()[0], config_.hidden_size}, DataType::Float32, DeviceType::CPU);
                char* dst = static_cast<char*>(gu_w->data());
                std::memcpy(dst, g_w->data(), g_size);
                std::memcpy(dst + g_size, u_w->data(), u_size);
                layers[i]->expert_gate_up[e]->weight = gu_w;
            }

            if (loader.has_tensor(ep + "down_proj.weight")) {
                layers[i]->expert_down[e]->weight = loader.load_tensor(ep + "down_proj.weight");
            }
        }
    }
}

std::shared_ptr<Tensor> Qwen3MoEForCausalLM::forward(const std::shared_ptr<Tensor>& input_ids,
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
