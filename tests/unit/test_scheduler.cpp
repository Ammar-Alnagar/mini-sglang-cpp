#include "minisgl/cache/kvcache.h"
#include "minisgl/model/qwen3.h"
#include "minisgl/scheduler/scheduler.h"
#include <cassert>
#include <iostream>

void test_kv_pool() {
    std::cout << "Testing KVCachePool..." << std::endl;
    minisgl::KVCachePool pool(10, 16, 8, 64, minisgl::DeviceType::CPU);
    assert(pool.total_pages() == 10);
    assert(pool.num_free_pages() == 10);

    int p0 = pool.allocate_page();
    int p1 = pool.allocate_page();
    assert(pool.num_free_pages() == 8);

    pool.free_page(p0);
    pool.free_page(p1);
    assert(pool.num_free_pages() == 10);
    std::cout << "KVCachePool test passed!" << std::endl;
}

void test_radix_cache() {
    std::cout << "Testing RadixCache..." << std::endl;
    auto pool = std::make_shared<minisgl::KVCachePool>(100, 1, 8, 64, minisgl::DeviceType::CPU);
    minisgl::RadixCache cache(pool);

    std::vector<int32_t> seq1 = {1, 2, 3, 4, 5};
    std::vector<int> pages1 = {10, 11, 12, 13, 14};

    cache.insert_prefix(seq1, pages1);

    auto match = cache.match_prefix(seq1);
    assert(match.matched_tokens == 5);
    assert(match.page_indices.size() == 5);

    std::vector<int32_t> seq2 = {1, 2, 3, 9, 10};
    auto match2 = cache.match_prefix(seq2);
    assert(match2.matched_tokens == 3);

    std::cout << "RadixCache test passed!" << std::endl;
}

void test_scheduler() {
    std::cout << "Testing Scheduler..." << std::endl;
    minisgl::ModelConfig cfg;
    cfg.num_layers = 2;
    cfg.hidden_size = 64;
    cfg.num_qo_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 1000;

    auto model = std::make_shared<minisgl::Qwen3ForCausalLM>(cfg);

    minisgl::SchedulerConfig sched_cfg;
    sched_cfg.max_pages = 100;
    sched_cfg.max_batch_size = 8;

    minisgl::Scheduler scheduler(sched_cfg, model);

    minisgl::SamplingParams params;
    params.max_tokens = 5;

    auto req1 = std::make_shared<minisgl::Req>(1, std::vector<int32_t>{1, 2, 3}, 0, 0, 5, params);
    scheduler.add_request(req1);

    assert(scheduler.has_pending_work());

    for (int i = 0; i < 6; ++i) {
        scheduler.step();
    }

    assert(req1->finished);
    assert(!scheduler.finished_reqs().empty());
    std::cout << "Scheduler test passed!" << std::endl;
}

int main() {
    test_kv_pool();
    test_radix_cache();
    test_scheduler();
    std::cout << "All scheduler unit tests passed successfully!" << std::endl;
    return 0;
}
