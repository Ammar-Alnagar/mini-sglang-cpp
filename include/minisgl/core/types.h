#ifndef MINISGL_CORE_TYPES_H
#define MINISGL_CORE_TYPES_H

#include "minisgl/core/tensor.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace minisgl {

struct SamplingParams {
    float temperature{0.0f};
    int top_k{-1};
    float top_p{1.0f};
    bool ignore_eos{false};
    int max_tokens{1024};

    bool is_greedy() const { return (temperature <= 0.0f || top_k == 1) && top_p == 1.0f; }
};

enum class ReqPhase { PREFILL, DECODE };

struct CacheHandle {
    int page_table_idx{-1};
    std::vector<int> page_indices;
};

class Req {
  public:
    int uid;
    int table_idx;
    int cached_len;
    int device_len;
    int max_device_len;
    int output_len;
    std::vector<int32_t> input_ids;
    SamplingParams sampling_params;
    CacheHandle cache_handle;
    bool finished{false};

    Req(int uid, std::vector<int32_t> input_tokens, int table_idx, int cached_len, int output_len,
        SamplingParams params)
        : uid(uid), table_idx(table_idx), cached_len(cached_len), output_len(output_len),
          input_ids(std::move(input_tokens)), sampling_params(params) {
        device_len = static_cast<int>(input_ids.size());
        max_device_len = device_len + output_len;
    }

    int remain_len() const { return max_device_len - device_len; }
    int extend_len() const { return device_len - cached_len; }
    bool can_decode() const { return remain_len() > 0 && !finished; }

    void complete_one() {
        cached_len = device_len;
        device_len += 1;
    }

    void append_token(int32_t token) { input_ids.push_back(token); }
};

struct Batch {
    ReqPhase phase{ReqPhase::PREFILL};
    std::vector<std::shared_ptr<Req>> reqs;
    std::vector<std::shared_ptr<Req>> padded_reqs;

    std::shared_ptr<Tensor> input_ids;
    std::shared_ptr<Tensor> positions;
    std::shared_ptr<Tensor> out_loc;

    bool is_prefill() const { return phase == ReqPhase::PREFILL; }
    bool is_decode() const { return phase == ReqPhase::DECODE; }
    size_t size() const { return reqs.size(); }
    size_t padded_size() const { return padded_reqs.size(); }
};

struct Context {
    int page_size{1};
    std::shared_ptr<Tensor> page_table;
    std::shared_ptr<Batch> current_batch;
};

void set_global_context(std::shared_ptr<Context> ctx);
std::shared_ptr<Context> get_global_context();

} // namespace minisgl

#endif // MINISGL_CORE_TYPES_H
