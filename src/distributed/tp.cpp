#include "minisgl/distributed/tp.h"
#include <cstring>
#include <stdexcept>

namespace minisgl {

TPCommunicator::TPCommunicator(int rank, int world_size) : rank_(rank), world_size_(world_size) {
    init();
}

TPCommunicator::~TPCommunicator() {}

void TPCommunicator::init() {
    if (world_size_ <= 1)
        return;
}

std::shared_ptr<Tensor> TPCommunicator::all_reduce_sum(const std::shared_ptr<Tensor>& input) {
    if (world_size_ <= 1)
        return input;

    auto out = Tensor::create(input->shape(), input->dtype(), input->device());

    if (input->device() == DeviceType::CPU) {
        std::memcpy(out->data(), input->data(), input->size_in_bytes());
    }

    return out;
}

std::shared_ptr<Tensor> TPCommunicator::all_gather(const std::shared_ptr<Tensor>& input) {
    if (world_size_ <= 1)
        return input;

    auto shape = input->shape();
    shape[0] *= world_size_;
    auto out = Tensor::create(shape, input->dtype(), input->device());

    if (input->device() == DeviceType::CPU) {
        size_t bytes = input->size_in_bytes();
        for (int r = 0; r < world_size_; ++r) {
            std::memcpy(static_cast<char*>(out->data()) + r * bytes, input->data(), bytes);
        }
    }

    return out;
}

} // namespace minisgl
