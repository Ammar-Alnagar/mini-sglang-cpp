#include "minisgl/model/config.h"
#include <fstream>
#include <stdexcept>

namespace minisgl {

ModelConfig ModelConfig::from_json(const nlohmann::json& j) {
    ModelConfig config;

    const nlohmann::json& root = (j.contains("text_config") && j["text_config"].is_object()) ? j["text_config"] : j;

    if (root.contains("num_hidden_layers")) config.num_layers = root["num_hidden_layers"].get<int>();
    if (root.contains("num_attention_heads")) config.num_qo_heads = root["num_attention_heads"].get<int>();
    if (root.contains("num_key_value_heads")) config.num_kv_heads = root["num_key_value_heads"].get<int>();
    else config.num_kv_heads = config.num_qo_heads;

    if (root.contains("hidden_size")) config.hidden_size = root["hidden_size"].get<int>();
    if (root.contains("vocab_size")) config.vocab_size = root["vocab_size"].get<int>();
    if (root.contains("intermediate_size")) config.intermediate_size = root["intermediate_size"].get<int>();
    if (root.contains("rms_norm_eps")) config.rms_norm_eps = root["rms_norm_eps"].get<float>();
    if (root.contains("hidden_act")) config.hidden_act = root["hidden_act"].get<std::string>();
    if (root.contains("tie_word_embeddings")) config.tie_word_embeddings = root["tie_word_embeddings"].get<bool>();
    if (root.contains("model_type")) config.model_type = root["model_type"].get<std::string>();

    if (root.contains("head_dim")) {
        config.head_dim = root["head_dim"].get<int>();
    } else {
        config.head_dim = config.hidden_size / config.num_qo_heads;
    }

    config.rotary_config.head_dim = config.head_dim;
    config.rotary_config.rotary_dim = config.head_dim;
    if (root.contains("max_position_embeddings")) {
        config.rotary_config.max_position = root["max_position_embeddings"].get<int>();
    }

    if (root.contains("rope_theta")) {
        config.rotary_config.base = root["rope_theta"].get<float>();
    } else if (root.contains("rope_scaling") && root["rope_scaling"].contains("rope_theta")) {
        config.rotary_config.base = root["rope_scaling"]["rope_theta"].get<float>();
    }

    // MoE fields
    if (root.contains("num_experts")) config.num_experts = root["num_experts"].get<int>();
    else if (root.contains("num_local_experts")) config.num_experts = root["num_local_experts"].get<int>();

    if (root.contains("num_experts_per_tok")) config.num_experts_per_tok = root["num_experts_per_tok"].get<int>();
    if (root.contains("moe_intermediate_size")) config.moe_intermediate_size = root["moe_intermediate_size"].get<int>();
    if (root.contains("norm_topk_prob")) config.norm_topk_prob = root["norm_topk_prob"].get<bool>();

    if (j.contains("architectures") && j["architectures"].is_array()) {
        for (const auto& arch : j["architectures"]) {
            config.architectures.push_back(arch.get<std::string>());
        }
    }

    return config;
}

ModelConfig ModelConfig::from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open model config file: " + path);
    }
    nlohmann::json j;
    f >> j;
    return from_json(j);
}

} // namespace minisgl
