#include <chrono>
#include <httplib.h>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 1919;

    std::cout << "Mini-SGLang Online API Benchmark targeting " << host << ":" << port << std::endl;

    httplib::Client cli(host, port);
    cli.set_connection_timeout(5, 0);

    auto start = std::chrono::high_resolution_clock::now();

    json req_body = {
        {"messages", json::array({{{"role", "user"}, {"content", "Benchmark prompt"}}})},
        {"max_tokens", 16},
        {"stream", false}};

    auto res = cli.Post("/v1/chat/completions", req_body.dump(), "application/json");

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

    if (res && res->status == 200) {
        std::cout << "Request successful! Latency: " << elapsed_ms << " ms" << std::endl;
    } else {
        std::cout << "Server not reachable or returned error (Ensure minisgl server is running)."
                  << std::endl;
    }

    return 0;
}
