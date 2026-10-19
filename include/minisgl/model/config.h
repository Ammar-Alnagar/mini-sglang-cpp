#ifndef MINISGL_MODEL_CONFIG_H
#define MINISGL_MODEL_CONFIG_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace minisgl {

struct RotaryConfig {
    int head_dim{128};
    int rotary_dim{128};
    int max_position{32768};
    float base{10000.0f};
};

struct ModelConfig {
    int num_layers{32};
    int num_qo_heads{32};
    int num_kv_heads{8};
    int head_dim{128};
    int hidden_size{4096};
    int vocab_size{151936};
    int intermediate_size{11008};
    float rms_norm_eps{1e-6f};
    RotaryConfig rotary_config;
    std::string hidden_act{"silu"};
    bool tie_word_embeddings{false};

    // MoE fields
    int num_experts{0};
    int num_experts_per_tok{0};
    int moe_intermediate_size{0};
    bool norm_topk_prob{false};

    std::string model_type{"qwen2"};
    std::vector<std::string> architectures;

    bool is_moe() const { return num_experts > 0 || model_type.find("moe") != std::string::npos; }

    static ModelConfig from_json(const nlohmann::json& j);
    static ModelConfig from_file(const std::string& path);
};

} // namespace minisgl

#endif // MINISGL_MODEL_CONFIG_H
