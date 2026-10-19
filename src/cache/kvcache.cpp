#include "minisgl/cache/kvcache.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace minisgl {

KVCachePool::KVCachePool(int num_pages, int page_size, int num_heads, int head_dim,
                         DeviceType device)
    : num_pages_(num_pages), page_size_(page_size), num_heads_(num_heads), head_dim_(head_dim) {
    k_cache =
        Tensor::create({num_pages * page_size, num_heads * head_dim}, DataType::Float32, device);
    v_cache =
        Tensor::create({num_pages * page_size, num_heads * head_dim}, DataType::Float32, device);

    for (int i = 0; i < num_pages; ++i) {
        free_pages_.push_back(i);
    }
}

int KVCachePool::allocate_page() {
    if (free_pages_.empty()) {
        throw std::runtime_error("KV Cache OOM: No free pages available");
    }
    int page = free_pages_.back();
    free_pages_.pop_back();
    return page;
}

void KVCachePool::free_page(int page_idx) {
    free_pages_.push_back(page_idx);
}

RadixCache::RadixCache(std::shared_ptr<KVCachePool> pool) : pool_(std::move(pool)) {
    root_ = std::make_shared<RadixNode>();
    root_->id = node_counter_++;
    root_->ref_count = 1;
}

RadixCache::MatchResult RadixCache::match_prefix(const std::vector<int32_t>& input_tokens) {
    MatchResult result;
    auto current = root_;
    size_t token_offset = 0;

    while (token_offset < input_tokens.size()) {
        int32_t next_tok = input_tokens[token_offset];
        auto it = current->children.find(next_tok);
        if (it == current->children.end())
            break;

        auto child = it->second;
        child->timestamp = ++current_time_;

        size_t match_count = 0;
        while (match_count < child->key_tokens.size() &&
               token_offset + match_count < input_tokens.size() &&
               child->key_tokens[match_count] == input_tokens[token_offset + match_count]) {
            match_count++;
        }

        result.matched_tokens += static_cast<int>(match_count);
        token_offset += match_count;

        for (size_t p = 0; p < child->value_pages.size() && p * 1 <= match_count; ++p) {
            result.page_indices.push_back(child->value_pages[p]);
        }

        if (match_count < child->key_tokens.size()) {
            break;
        }

        current = child;
    }

    result.node = current;
    return result;
}

void RadixCache::insert_prefix(const std::vector<int32_t>& input_tokens,
                               const std::vector<int>& page_indices) {
    if (input_tokens.empty())
        return;

    auto node = std::make_shared<RadixNode>();
    node->id = node_counter_++;
    node->key_tokens = input_tokens;
    node->value_pages = page_indices;
    node->parent = root_;
    node->timestamp = ++current_time_;

    root_->children[input_tokens[0]] = node;
}

void RadixCache::evict(int needed_pages) {
    std::vector<std::shared_ptr<RadixNode>> leaves;
    for (auto& [tok, child] : root_->children) {
        if (child->is_leaf() && child->ref_count == 0) {
            leaves.push_back(child);
        }
    }

    std::sort(leaves.begin(), leaves.end(),
              [](const auto& a, const auto& b) { return a->timestamp < b->timestamp; });

    int freed = 0;
    for (auto& leaf : leaves) {
        if (freed >= needed_pages)
            break;
        for (int p : leaf->value_pages) {
            pool_->free_page(p);
            freed++;
        }
        root_->children.erase(leaf->key_tokens[0]);
    }
}

NaiveCache::NaiveCache(std::shared_ptr<KVCachePool> pool) : pool_(std::move(pool)) {}

std::vector<int> NaiveCache::allocate(int num_tokens) {
    std::vector<int> pages;
    for (int i = 0; i < num_tokens; ++i) {
        pages.push_back(pool_->allocate_page());
    }
    return pages;
}

void NaiveCache::free(const std::vector<int>& page_indices) {
    for (int p : page_indices) {
        pool_->free_page(p);
    }
}

} // namespace minisgl
