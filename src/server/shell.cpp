#include "minisgl/server/shell.h"
#include <iostream>
#include <string>

namespace minisgl {

InteractiveShell::InteractiveShell(std::shared_ptr<Tokenizer> tokenizer, std::shared_ptr<Scheduler> scheduler)
    : tokenizer_(std::move(tokenizer)), scheduler_(std::move(scheduler)) {}

void InteractiveShell::run() {
    std::cout << "Mini-SGLang Interactive Shell" << std::endl;
    std::cout << "Type /reset to clear chat history, or Ctrl+C to exit." << std::endl;

    std::string line;
    while (true) {
        std::cout << "\nUser: ";
        if (!std::getline(std::cin, line)) break;

        if (line == "/reset") {
            history_.clear();
            std::cout << "Chat history reset." << std::endl;
            continue;
        }

        if (line.empty()) continue;

        history_.push_back({"user", line});
        std::string prompt = tokenizer_->apply_chat_template(history_);
        auto input_tokens = tokenizer_->encode(prompt);

        SamplingParams params;
        params.max_tokens = 512;

        auto req = std::make_shared<Req>(0, input_tokens, 0, 0, params.max_tokens, params);
        scheduler_->add_request(req);

        std::cout << "Assistant: " << std::flush;
        std::string assistant_reply;

        while (!req->finished) {
            scheduler_->step();
            if (req->input_ids.size() > input_tokens.size()) {
                int32_t last_tok = req->input_ids.back();
                std::string chunk = tokenizer_->detokenize_incremental(req->uid, last_tok, req->finished);
                std::cout << chunk << std::flush;
                assistant_reply += chunk;
            }
        }
        std::cout << std::endl;
        history_.push_back({"assistant", assistant_reply});
    }
}

} // namespace minisgl
