#include "minisgl/tokenizer/tokenizer.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>
#include <algorithm>

namespace minisgl {

namespace fs = std::filesystem;

Tokenizer::Tokenizer() = default;

Tokenizer::Tokenizer(const std::string& model_dir) {
    load(model_dir);
}

void Tokenizer::parse_tokenizer_config_json(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) return;
    try {
        nlohmann::json j;
        f >> j;
        if (j.contains("eos_token_id") && j["eos_token_id"].is_number_integer()) {
            eos_token_id_ = j["eos_token_id"].get<int32_t>();
        }
        if (j.contains("bos_token_id") && j["bos_token_id"].is_number_integer()) {
            bos_token_id_ = j["bos_token_id"].get<int32_t>();
        }
    } catch (...) {}
}

void Tokenizer::parse_tokenizer_json(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) return;
    try {
        nlohmann::json j;
        f >> j;
        if (j.contains("model") && j["model"].contains("vocab")) {
            auto vocab = j["model"]["vocab"];
            for (auto& [tok, id] : vocab.items()) {
                int32_t tid = id.get<int32_t>();
                token_to_id_[tok] = tid;
                id_to_token_[tid] = tok;
            }
        }
        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (auto& item : j["added_tokens"]) {
                if (item.contains("id") && item.contains("content")) {
                    int32_t tid = item["id"].get<int32_t>();
                    std::string content = item["content"].get<std::string>();
                    token_to_id_[content] = tid;
                    id_to_token_[tid] = content;
                }
            }
        }
    } catch (...) {}
}

bool Tokenizer::load(const std::string& model_dir) {
    std::string tok_json = (fs::path(model_dir) / "tokenizer.json").string();
    std::string tok_cfg = (fs::path(model_dir) / "tokenizer_config.json").string();

    parse_tokenizer_json(tok_json);
    parse_tokenizer_config_json(tok_cfg);

    return !token_to_id_.empty();
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages) const {
    std::string formatted;
    for (const auto& msg : messages) {
        formatted += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    formatted += "<|im_start|>assistant\n";
    return formatted;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens;
    if (token_to_id_.empty()) {
        for (char c : text) {
            tokens.push_back(static_cast<uint8_t>(c));
        }
        return tokens;
    }

    size_t i = 0;
    while (i < text.length()) {
        bool matched = false;
        size_t max_len = std::min(text.length() - i, static_cast<size_t>(64));
        for (size_t len = max_len; len > 0; --len) {
            std::string sub = text.substr(i, len);
            auto it = token_to_id_.find(sub);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            std::string byte_str(1, text[i]);
            auto it = token_to_id_.find(byte_str);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(static_cast<uint8_t>(text[i]));
            }
            i += 1;
        }
    }
    return tokens;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    if (token_id >= 0 && token_id < 256) {
        return std::string(1, static_cast<char>(token_id));
    }
    return "";
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string text;
    for (int32_t tid : tokens) {
        text += decode_token(tid);
    }
    return text;
}

std::string Tokenizer::detokenize_incremental(int uid, int32_t next_token, bool finished) {
    if (decode_map_.find(uid) == decode_map_.end()) {
        decode_map_[uid] = DecodeStatus();
    }

    auto& status = decode_map_[uid];
    if (!finished || next_token != eos_token_id_) {
        status.decoded_ids.push_back(next_token);
    }

    std::string new_str = decode_token(next_token);
    status.decoded_str += new_str;

    std::string output = status.decoded_str.substr(status.sent_offset);
    status.sent_offset = status.decoded_str.length();

    if (finished) {
        decode_map_.erase(uid);
    }

    return output;
}

} // namespace minisgl
