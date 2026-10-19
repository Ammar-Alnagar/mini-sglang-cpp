#ifndef MINISGL_DISTRIBUTED_TP_H
#define MINISGL_DISTRIBUTED_TP_H

#include "minisgl/core/tensor.h"
#include <memory>

namespace minisgl {

class TPCommunicator {
  public:
    TPCommunicator(int rank = 0, int world_size = 1);
    ~TPCommunicator();

    void init();
    std::shared_ptr<Tensor> all_reduce_sum(const std::shared_ptr<Tensor>& input);
    std::shared_ptr<Tensor> all_gather(const std::shared_ptr<Tensor>& input);

    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

  private:
    int rank_{0};
    int world_size_{1};
    void* nccl_comm_{nullptr};
};

} // namespace minisgl

#endif // MINISGL_DISTRIBUTED_TP_H
