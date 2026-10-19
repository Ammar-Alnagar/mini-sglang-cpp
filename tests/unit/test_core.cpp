#include "minisgl/core/tensor.h"
#include "minisgl/core/types.h"
#include "minisgl/model/config.h"
#include "minisgl/tokenizer/tokenizer.h"
#include <iostream>
#include <cassert>

void test_tensor() {
    std::cout << "Testing Tensor..." << std::endl;
    auto t = minisgl::Tensor::create({2, 4}, minisgl::DataType::Float32, minisgl::DeviceType::CPU);
    assert(t->numel() == 8);
    assert(t->size_in_bytes() == 32);

    float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    t->copy_from(data, sizeof(data), minisgl::DeviceType::CPU);

    float read_data[8] = {0};
    t->copy_to(read_data, sizeof(read_data), minisgl::DeviceType::CPU);
    for (int i = 0; i < 8; ++i) {
        assert(read_data[i] == data[i]);
    }
    std::cout << "Tensor test passed!" << std::endl;
}

void test_config() {
    std::cout << "Testing ModelConfig..." << std::endl;
    nlohmann::json j = {
        {"architectures", {"Qwen2ForCausalLM"}},
        {"hidden_size", 1024},
        {"num_attention_heads", 16},
        {"num_key_value_heads", 4},
        {"num_hidden_layers", 12},
        {"vocab_size", 151936},
        {"intermediate_size", 2816},
        {"rms_norm_eps", 1e-6},
        {"model_type", "qwen2"}
    };
    auto cfg = minisgl::ModelConfig::from_json(j);
    assert(cfg.hidden_size == 1024);
    assert(cfg.num_qo_heads == 16);
    assert(cfg.num_kv_heads == 4);
    assert(cfg.num_layers == 12);
    assert(cfg.head_dim == 64);
    assert(!cfg.is_moe());
    std::cout << "ModelConfig test passed!" << std::endl;
}

void test_tokenizer() {
    std::cout << "Testing Tokenizer..." << std::endl;
    minisgl::Tokenizer tok;
    auto tokens = tok.encode("Hello World");
    assert(!tokens.empty());
    auto decoded = tok.decode(tokens);
    assert(decoded == "Hello World");

    auto stream_str = tok.detokenize_incremental(1, tokens[0], false);
    assert(!stream_str.empty());
    std::cout << "Tokenizer test passed!" << std::endl;
}

int main() {
    test_tensor();
    test_config();
    test_tokenizer();
    std::cout << "All core unit tests passed successfully!" << std::endl;
    return 0;
}
