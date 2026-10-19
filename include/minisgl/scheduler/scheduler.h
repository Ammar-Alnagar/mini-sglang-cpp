#ifndef MINISGL_SCHEDULER_SCHEDULER_H
#define MINISGL_SCHEDULER_SCHEDULER_H

#include "minisgl/cache/kvcache.h"
#include "minisgl/core/types.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/model/qwen3_moe.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace minisgl {

struct SchedulerConfig {
    int max_prefill_len{1024};
    int max_batch_size{128};
    int page_size{1};
    int max_pages{1000};
    bool use_radix_cache{true};
    bool overlap_scheduling{true};
};

class Scheduler {
  public:
    Scheduler(SchedulerConfig config, std::shared_ptr<Qwen3ForCausalLM> model);

    void add_request(std::shared_ptr<Req> req);
    bool has_pending_work() const;

    std::shared_ptr<Batch> schedule_next_batch();
    void step();

    void run_loop();
    void stop();

    const std::vector<std::shared_ptr<Req>>& finished_reqs() const { return finished_reqs_; }

  private:
    SchedulerConfig config_;
    std::shared_ptr<Qwen3ForCausalLM> model_;
    std::shared_ptr<KVCachePool> pool_;
    std::shared_ptr<RadixCache> radix_cache_;
    std::shared_ptr<NaiveCache> naive_cache_;

    std::queue<std::shared_ptr<Req>> waiting_queue_;
    std::vector<std::shared_ptr<Req>> running_reqs_;
    std::vector<std::shared_ptr<Req>> finished_reqs_;

    mutable std::mutex queue_mutex_;
    std::atomic<bool> running_{false};
    int req_id_counter_{0};
};

} // namespace minisgl

#endif // MINISGL_SCHEDULER_SCHEDULER_H
