#include "minisgl/model/config.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/server/server.h"
#include <cassert>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thread>

using json = nlohmann::json;

void test_server() {
    std::cout << "Testing HttpServer endpoints..." << std::endl;

    minisgl::ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_qo_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 1000;

    auto model = std::make_shared<minisgl::Qwen3ForCausalLM>(cfg);
    auto tokenizer = std::make_shared<minisgl::Tokenizer>();

    minisgl::SchedulerConfig sched_cfg;
    sched_cfg.max_pages = 100;
    auto scheduler = std::make_shared<minisgl::Scheduler>(sched_cfg, model);

    int port = 19199;
    auto server = std::make_shared<minisgl::HttpServer>("127.0.0.1", port, tokenizer, scheduler);

    std::thread server_thread([server]() { server->start(); });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    httplib::Client cli("127.0.0.1", port);

    auto res_health = cli.Get("/health");
    assert(res_health && res_health->status == 200);
    json j_health = json::parse(res_health->body);
    assert(j_health["status"] == "ok");

    json req_body = {{"messages", json::array({{{"role", "user"}, {"content", "Hello"}}})},
                     {"max_tokens", 4},
                     {"stream", false}};

    auto res_chat = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");
    assert(res_chat && res_chat->status == 200);
    json j_chat = json::parse(res_chat->body);
    assert(j_chat.contains("choices"));
    assert(!j_chat["choices"].empty());

    server->stop();
    server_thread.join();

    std::cout << "HttpServer test passed!" << std::endl;
}

int main() {
    test_server();
    return 0;
}
