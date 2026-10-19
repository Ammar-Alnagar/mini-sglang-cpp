#include "minisgl/server/args.h"
#include "minisgl/server/server.h"
#include "minisgl/server/shell.h"
#include "minisgl/tokenizer/tokenizer.h"
#include "minisgl/model/config.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/model/qwen3_moe.h"
#include "minisgl/model/weight.h"
#include "minisgl/scheduler/scheduler.h"
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    auto args = minisgl::CliArgs::parse(argc, argv);

    std::cout << "Mini-SGLang Native C++/CUDA Inference Engine" << std::endl;
    std::cout << "Model: " << args.model << std::endl;

    auto tokenizer = std::make_shared<minisgl::Tokenizer>();
    if (fs::exists(args.model)) {
        tokenizer->load(args.model);
    }

    minisgl::ModelConfig config;
    std::string config_path = (fs::path(args.model) / "config.json").string();
    if (fs::exists(config_path)) {
        config = minisgl::ModelConfig::from_file(config_path);
    } else {
        std::cout << "Config file not found at " << config_path << ", using default configuration." << std::endl;
        config.num_layers = 4;
        config.hidden_size = 256;
        config.num_qo_heads = 8;
        config.num_kv_heads = 4;
        config.head_dim = 32;
        config.intermediate_size = 512;
        config.vocab_size = 151936;
    }

    auto model = std::make_shared<minisgl::Qwen3ForCausalLM>(config);

    if (fs::exists(args.model)) {
        minisgl::WeightLoader loader(args.model);
        model->load_weights(loader);
    }

    minisgl::SchedulerConfig sched_config;
    sched_config.max_prefill_len = args.max_prefill_length;
    sched_config.page_size = args.page_size;
    sched_config.use_radix_cache = (args.cache == "radix");

    auto scheduler = std::make_shared<minisgl::Scheduler>(sched_config, model);

    if (args.shell) {
        minisgl::InteractiveShell shell(tokenizer, scheduler);
        shell.run();
    } else {
        minisgl::HttpServer server(args.host, args.port, tokenizer, scheduler);
        server.start();
    }

    return 0;
}
