#include "minisgl/model/config.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/scheduler/scheduler.h"
#include <chrono>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Mini-SGLang Offline Benchmark" << std::endl;

    minisgl::ModelConfig cfg;
    cfg.num_layers = 4;
    cfg.hidden_size = 256;
    cfg.num_qo_heads = 8;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 32;
    cfg.intermediate_size = 512;
    cfg.vocab_size = 10000;

    auto model = std::make_shared<minisgl::Qwen3ForCausalLM>(cfg);

    minisgl::SchedulerConfig sched_cfg;
    sched_cfg.max_pages = 1000;
    sched_cfg.max_batch_size = 16;

    minisgl::Scheduler scheduler(sched_cfg, model);

    int num_requests = 32;
    int prompt_len = 128;
    int gen_len = 32;

    std::vector<int32_t> prompt(prompt_len, 1);
    minisgl::SamplingParams params;
    params.max_tokens = gen_len;

    for (int i = 0; i < num_requests; ++i) {
        auto req = std::make_shared<minisgl::Req>(i, prompt, 0, 0, gen_len, params);
        scheduler.add_request(req);
    }

    auto start = std::chrono::high_resolution_clock::now();

    while (scheduler.finished_reqs().size() < static_cast<size_t>(num_requests)) {
        scheduler.step();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(end - start).count();

    int total_gen_tokens = num_requests * gen_len;
    std::cout << "Total requests: " << num_requests << std::endl;
    std::cout << "Elapsed time: " << elapsed_sec << " seconds" << std::endl;
    std::cout << "Throughput: " << total_gen_tokens / elapsed_sec << " tokens/sec" << std::endl;

    return 0;
}
