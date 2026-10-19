#ifndef MINISGL_CORE_TENSOR_H
#define MINISGL_CORE_TENSOR_H

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <iostream>

namespace minisgl {

enum class DataType {
    Float32,
    Float16,
    BFloat16,
    Int32,
    Int64,
    Int8
};

enum class DeviceType {
    CPU,
    CUDA
};

size_t get_dtype_size(DataType dtype);

class Tensor {
public:
    Tensor();
    Tensor(std::vector<int64_t> shape, DataType dtype, DeviceType device = DeviceType::CPU);
    ~Tensor();

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    static std::shared_ptr<Tensor> create(std::vector<int64_t> shape, DataType dtype, DeviceType device = DeviceType::CPU);

    const std::vector<int64_t>& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    DeviceType device() const { return device_; }
    void* data() { return data_; }
    const void* data() const { return data_; }

    template<typename T>
    T* data_ptr() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data_); }

    int64_t numel() const;
    size_t size_in_bytes() const;

    void allocate();
    void free();
    void copy_from(const void* src, size_t size_bytes, DeviceType src_device = DeviceType::CPU);
    void copy_to(void* dst, size_t size_bytes, DeviceType dst_device = DeviceType::CPU) const;

    std::shared_ptr<Tensor> to(DeviceType new_device) const;

private:
    std::vector<int64_t> shape_;
    DataType dtype_;
    DeviceType device_;
    void* data_{nullptr};
    bool owns_data_{true};
};

} // namespace minisgl

#endif // MINISGL_CORE_TENSOR_H
