#include "minisgl/scheduler/scheduler.h"
#include <algorithm>
#include <iostream>

namespace minisgl {

Scheduler::Scheduler(SchedulerConfig config, std::shared_ptr<Qwen3ForCausalLM> model)
    : config_(config), model_(std::move(model)) {
    pool_ =
        std::make_shared<KVCachePool>(config.max_pages, config.page_size, 8, 128, DeviceType::CPU);
    radix_cache_ = std::make_shared<RadixCache>(pool_);
    naive_cache_ = std::make_shared<NaiveCache>(pool_);
}

void Scheduler::add_request(std::shared_ptr<Req> req) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    req->uid = req_id_counter_++;
    waiting_queue_.push(req);
}

bool Scheduler::has_pending_work() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return !waiting_queue_.empty() || !running_reqs_.empty();
}

std::shared_ptr<Batch> Scheduler::schedule_next_batch() {
    std::lock_guard<std::mutex> lock(queue_mutex_);

    auto batch = std::make_shared<Batch>();

    if (!waiting_queue_.empty()) {
        batch->phase = ReqPhase::PREFILL;
        int prefill_tokens = 0;

        while (!waiting_queue_.empty() &&
               batch->reqs.size() < static_cast<size_t>(config_.max_batch_size)) {
            auto req = waiting_queue_.front();
            if (prefill_tokens + req->device_len > config_.max_prefill_len &&
                !batch->reqs.empty()) {
                break;
            }

            waiting_queue_.pop();

            if (config_.use_radix_cache) {
                auto match = radix_cache_->match_prefix(req->input_ids);
                req->cached_len = match.matched_tokens;
                req->cache_handle.page_indices = match.page_indices;
            }

            int needed_pages = req->extend_len();
            auto pages = naive_cache_->allocate(needed_pages);
            req->cache_handle.page_indices.insert(req->cache_handle.page_indices.end(),
                                                  pages.begin(), pages.end());

            prefill_tokens += req->extend_len();
            batch->reqs.push_back(req);
            running_reqs_.push_back(req);
        }
        return batch;
    }

    if (!running_reqs_.empty()) {
        batch->phase = ReqPhase::DECODE;
        for (auto& req : running_reqs_) {
            if (req->can_decode()) {
                auto pages = naive_cache_->allocate(1);
                req->cache_handle.page_indices.push_back(pages[0]);
                batch->reqs.push_back(req);
            }
        }
        return batch;
    }

    return nullptr;
}

void Scheduler::step() {
    auto batch = schedule_next_batch();
    if (!batch || batch->reqs.empty())
        return;

    std::vector<int32_t> batch_input_ids;
    std::vector<int32_t> batch_positions;

    for (auto& req : batch->reqs) {
        if (batch->is_prefill()) {
            for (int i = req->cached_len; i < req->device_len; ++i) {
                batch_input_ids.push_back(req->input_ids[i]);
                batch_positions.push_back(i);
            }
        } else {
            batch_input_ids.push_back(req->input_ids.back());
            batch_positions.push_back(req->device_len - 1);
        }
    }

    auto in_ids_tensor = Tensor::create({static_cast<int64_t>(batch_input_ids.size())},
                                        DataType::Int32, DeviceType::CPU);
    in_ids_tensor->copy_from(batch_input_ids.data(), batch_input_ids.size() * sizeof(int32_t),
                             DeviceType::CPU);

    auto pos_tensor = Tensor::create({static_cast<int64_t>(batch_positions.size())},
                                     DataType::Int32, DeviceType::CPU);
    pos_tensor->copy_from(batch_positions.data(), batch_positions.size() * sizeof(int32_t),
                          DeviceType::CPU);

    auto logits = model_->forward(in_ids_tensor, pos_tensor);

    const float* logit_ptr = logits->data_ptr<float>();
    int vocab_size = static_cast<int>(logits->shape()[1]);

    for (size_t i = 0; i < batch->reqs.size(); ++i) {
        auto& req = batch->reqs[i];
        const float* req_logits = logit_ptr + i * vocab_size;

        int32_t best_token = 0;
        float max_val = req_logits[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (req_logits[v] > max_val) {
                max_val = req_logits[v];
                best_token = v;
            }
        }

        req->append_token(best_token);
        req->complete_one();

        if (req->device_len >= req->max_device_len || req->finished) {
            req->finished = true;
            finished_reqs_.push_back(req);

            naive_cache_->free(req->cache_handle.page_indices);
        }
    }

    running_reqs_.erase(std::remove_if(running_reqs_.begin(), running_reqs_.end(),
                                       [](const auto& r) { return r->finished; }),
                        running_reqs_.end());
}

void Scheduler::run_loop() {
    running_ = true;
    while (running_) {
        if (has_pending_work()) {
            step();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void Scheduler::stop() {
    running_ = false;
}

} // namespace minisgl
