#include "minisgl/distributed/tp.h"
#include <cassert>
#include <iostream>

void test_tp_communicator() {
    std::cout << "Testing TPCommunicator..." << std::endl;
    minisgl::TPCommunicator comm(0, 1);
    assert(comm.rank() == 0);
    assert(comm.world_size() == 1);

    auto t = minisgl::Tensor::create({4, 16}, minisgl::DataType::Float32, minisgl::DeviceType::CPU);
    auto res = comm.all_reduce_sum(t);
    assert(res->shape() == t->shape());

    auto gathered = comm.all_gather(t);
    assert(gathered->shape()[0] == 4);
    std::cout << "TPCommunicator test passed!" << std::endl;
}

int main() {
    test_tp_communicator();
    std::cout << "All distributed unit tests passed successfully!" << std::endl;
    return 0;
}
