#ifndef MINISGL_TOKENIZER_TOKENIZER_H
#define MINISGL_TOKENIZER_TOKENIZER_H

#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace minisgl {

struct ChatMessage {
    std::string role;
    std::string content;
};

struct DecodeStatus {
    std::vector<int32_t> decoded_ids;
    std::string decoded_str;
    size_t read_offset{0};
    size_t surr_offset{0};
    size_t sent_offset{0};
};

class Tokenizer {
  public:
    Tokenizer();
    explicit Tokenizer(const std::string& model_dir);

    bool load(const std::string& model_dir);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;

    std::string decode_token(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& tokens) const;

    std::string detokenize_incremental(int uid, int32_t next_token, bool finished);

    int32_t bos_token_id() const { return bos_token_id_; }
    int32_t eos_token_id() const { return eos_token_id_; }
    int32_t pad_token_id() const { return pad_token_id_; }
    int vocab_size() const { return static_cast<int>(token_to_id_.size()); }

  private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::unordered_map<int, DecodeStatus> decode_map_;

    int32_t bos_token_id_{-1};
    int32_t eos_token_id_{151643};
    int32_t pad_token_id_{151643};

    void parse_tokenizer_json(const std::string& filepath);
    void parse_tokenizer_config_json(const std::string& filepath);
};

} // namespace minisgl

#endif // MINISGL_TOKENIZER_TOKENIZER_H
