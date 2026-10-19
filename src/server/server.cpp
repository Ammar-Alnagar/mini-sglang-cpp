#include "minisgl/server/server.h"
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>

namespace minisgl {

using json = nlohmann::json;

HttpServer::HttpServer(std::string host, int port, std::shared_ptr<Tokenizer> tokenizer,
                       std::shared_ptr<Scheduler> scheduler)
    : host_(std::move(host)), port_(port), tokenizer_(std::move(tokenizer)),
      scheduler_(std::move(scheduler)) {
    setup_routes();
}

void HttpServer::setup_routes() {
    svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json j = {{"status", "ok"}};
        res.set_content(j.dump(), "application/json");
    });

    svr_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_chat_completions(req, res);
    });

    svr_.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        handle_completions(req, res);
    });
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        std::vector<ChatMessage> chat_msgs;

        if (body.contains("messages") && body["messages"].is_array()) {
            for (auto& item : body["messages"]) {
                chat_msgs.push_back(
                    {item["role"].get<std::string>(), item["content"].get<std::string>()});
            }
        }

        std::string prompt = tokenizer_->apply_chat_template(chat_msgs);
        auto tokens = tokenizer_->encode(prompt);

        SamplingParams params;
        if (body.contains("max_tokens"))
            params.max_tokens = body["max_tokens"].get<int>();
        if (body.contains("temperature"))
            params.temperature = body["temperature"].get<float>();

        bool is_stream = body.contains("stream") && body["stream"].get<bool>();

        auto req_obj = std::make_shared<Req>(0, tokens, 0, 0, params.max_tokens, params);
        scheduler_->add_request(req_obj);

        if (is_stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, req_obj, tokens](size_t offset, httplib::DataSink& sink) {
                    if (req_obj->finished) {
                        std::string done_msg = "data: [DONE]\n\n";
                        sink.write(done_msg.data(), done_msg.size());
                        sink.done();
                        return true;
                    }

                    scheduler_->step();
                    if (req_obj->input_ids.size() > tokens.size()) {
                        int32_t last_tok = req_obj->input_ids.back();
                        std::string chunk_str = tokenizer_->detokenize_incremental(
                            req_obj->uid, last_tok, req_obj->finished);
                        if (!chunk_str.empty()) {
                            json chunk_json = {
                                {"id", "chatcmpl-" + std::to_string(req_obj->uid)},
                                {"object", "chat.completion.chunk"},
                                {"created", std::time(nullptr)},
                                {"choices",
                                 json::array(
                                     {{{"index", 0},
                                       {"delta", {{"content", chunk_str}}},
                                       {"finish_reason", req_obj->finished ? "stop" : nullptr}}})}};
                            std::string msg = "data: " + chunk_json.dump() + "\n\n";
                            sink.write(msg.data(), msg.size());
                        }
                    }
                    return true;
                });
        } else {
            while (!req_obj->finished) {
                scheduler_->step();
            }
            std::vector<int32_t> gen_tokens(req_obj->input_ids.begin() + tokens.size(),
                                            req_obj->input_ids.end());
            std::string text = tokenizer_->decode(gen_tokens);

            json response = {
                {"id", "chatcmpl-" + std::to_string(req_obj->uid)},
                {"object", "chat.completion"},
                {"created", std::time(nullptr)},
                {"choices", json::array({{{"index", 0},
                                          {"message", {{"role", "assistant"}, {"content", text}}},
                                          {"finish_reason", "stop"}}})}};
            res.set_content(response.dump(), "application/json");
        }
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void HttpServer::handle_completions(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        std::string prompt = body.value("prompt", "");
        auto tokens = tokenizer_->encode(prompt);

        SamplingParams params;
        if (body.contains("max_tokens"))
            params.max_tokens = body["max_tokens"].get<int>();

        auto req_obj = std::make_shared<Req>(0, tokens, 0, 0, params.max_tokens, params);
        scheduler_->add_request(req_obj);

        while (!req_obj->finished) {
            scheduler_->step();
        }

        std::vector<int32_t> gen_tokens(req_obj->input_ids.begin() + tokens.size(),
                                        req_obj->input_ids.end());
        std::string text = tokenizer_->decode(gen_tokens);

        json response = {
            {"id", "cmpl-" + std::to_string(req_obj->uid)},
            {"object", "text_completion"},
            {"created", std::time(nullptr)},
            {"choices", json::array({{{"text", text}, {"index", 0}, {"finish_reason", "stop"}}})}};
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        res.status = 400;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void HttpServer::start() {
    std::cout << "Starting Mini-SGLang HTTP Server on " << host_ << ":" << port_ << std::endl;
    svr_.listen(host_.c_str(), port_);
}

void HttpServer::stop() {
    svr_.stop();
}

} // namespace minisgl
