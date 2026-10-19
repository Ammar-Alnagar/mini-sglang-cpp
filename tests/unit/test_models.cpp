#include "minisgl/model/config.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/model/qwen3_moe.h"
#include "minisgl/core/tensor.h"
#include <iostream>
#include <cassert>

void test_qwen3_forward() {
    std::cout << "Testing Qwen3 model forward..." << std::endl;
    minisgl::ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_qo_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 1000;

    minisgl::Qwen3ForCausalLM model(cfg);

    auto input_ids = minisgl::Tensor::create({4}, minisgl::DataType::Int32, minisgl::DeviceType::CPU);
    int32_t ids[4] = {10, 20, 30, 40};
    input_ids->copy_from(ids, sizeof(ids), minisgl::DeviceType::CPU);

    auto positions = minisgl::Tensor::create({4}, minisgl::DataType::Int32, minisgl::DeviceType::CPU);
    int32_t pos[4] = {0, 1, 2, 3};
    positions->copy_from(pos, sizeof(pos), minisgl::DeviceType::CPU);

    auto logits = model.forward(input_ids, positions);
    assert(logits != nullptr);
    assert(logits->shape()[0] == 4);
    assert(logits->shape()[1] == 1000);
    std::cout << "Qwen3 model forward test passed!" << std::endl;
}

void test_qwen3_moe_forward() {
    std::cout << "Testing Qwen3-MoE model forward..." << std::endl;
    minisgl::ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_qo_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 1000;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.moe_intermediate_size = 64;

    minisgl::Qwen3MoEForCausalLM model(cfg);

    auto input_ids = minisgl::Tensor::create({2}, minisgl::DataType::Int32, minisgl::DeviceType::CPU);
    int32_t ids[2] = {5, 15};
    input_ids->copy_from(ids, sizeof(ids), minisgl::DeviceType::CPU);

    auto positions = minisgl::Tensor::create({2}, minisgl::DataType::Int32, minisgl::DeviceType::CPU);
    int32_t pos[2] = {0, 1};
    positions->copy_from(pos, sizeof(pos), minisgl::DeviceType::CPU);

    auto logits = model.forward(input_ids, positions);
    assert(logits != nullptr);
    assert(logits->shape()[0] == 2);
    assert(logits->shape()[1] == 1000);
    std::cout << "Qwen3-MoE model forward test passed!" << std::endl;
}

int main() {
    test_qwen3_forward();
    test_qwen3_moe_forward();
    std::cout << "All model unit tests passed successfully!" << std::endl;
    return 0;
}
