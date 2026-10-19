#ifndef MINISGL_CACHE_KVCACHE_H
#define MINISGL_CACHE_KVCACHE_H

#include "minisgl/core/tensor.h"
#include "minisgl/core/types.h"
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace minisgl {

class KVCachePool {
  public:
    KVCachePool(int num_pages, int page_size, int num_heads, int head_dim,
                DeviceType device = DeviceType::CPU);

    int allocate_page();
    void free_page(int page_idx);

    int num_free_pages() const { return static_cast<int>(free_pages_.size()); }
    int total_pages() const { return num_pages_; }

    std::shared_ptr<Tensor> k_cache;
    std::shared_ptr<Tensor> v_cache;

  private:
    int num_pages_;
    int page_size_;
    int num_heads_;
    int head_dim_;
    std::vector<int> free_pages_;
};

struct RadixNode {
    int id;
    std::vector<int32_t> key_tokens;
    std::vector<int> value_pages;
    int ref_count{0};
    uint64_t timestamp{0};
    std::shared_ptr<RadixNode> parent{nullptr};
    std::map<int32_t, std::shared_ptr<RadixNode>> children;

    bool is_root() const { return parent == nullptr; }
    bool is_leaf() const { return children.empty(); }
};

class RadixCache {
  public:
    explicit RadixCache(std::shared_ptr<KVCachePool> pool);

    struct MatchResult {
        int matched_tokens{0};
        std::vector<int> page_indices;
        std::shared_ptr<RadixNode> node;
    };

    MatchResult match_prefix(const std::vector<int32_t>& input_tokens);
    void insert_prefix(const std::vector<int32_t>& input_tokens,
                       const std::vector<int>& page_indices);
    void evict(int needed_pages);

  private:
    std::shared_ptr<KVCachePool> pool_;
    std::shared_ptr<RadixNode> root_;
    int node_counter_{0};
    uint64_t current_time_{0};
};

class NaiveCache {
  public:
    explicit NaiveCache(std::shared_ptr<KVCachePool> pool);

    std::vector<int> allocate(int num_tokens);
    void free(const std::vector<int>& page_indices);

  private:
    std::shared_ptr<KVCachePool> pool_;
};

} // namespace minisgl

#endif // MINISGL_CACHE_KVCACHE_H
