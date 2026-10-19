#include "minisgl/core/tensor.h"
#include <cstring>
#include <numeric>
#include <stdexcept>

#ifdef MINISGL_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace minisgl {

size_t get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::BFloat16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::Int8: return 1;
        default: return 4;
    }
}

Tensor::Tensor() : dtype_(DataType::Float32), device_(DeviceType::CPU), data_(nullptr), owns_data_(false) {}

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, DeviceType device)
    : shape_(std::move(shape)), dtype_(dtype), device_(device) {
    allocate();
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      device_(other.device_),
      data_(other.data_),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

std::shared_ptr<Tensor> Tensor::create(std::vector<int64_t> shape, DataType dtype, DeviceType device) {
    return std::make_shared<Tensor>(std::move(shape), dtype, device);
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    int64_t count = 1;
    for (auto dim : shape_) {
        count *= dim;
    }
    return count;
}

size_t Tensor::size_in_bytes() const {
    return numel() * get_dtype_size(dtype_);
}

void Tensor::allocate() {
    size_t bytes = size_in_bytes();
    if (bytes == 0) {
        data_ = nullptr;
        return;
    }

    if (device_ == DeviceType::CPU) {
        data_ = std::malloc(bytes);
        if (!data_) {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
        std::memset(data_, 0, bytes);
        owns_data_ = true;
    } else if (device_ == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
        cudaError_t err = cudaMalloc(&data_, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        cudaMemset(data_, 0, bytes);
        owns_data_ = true;
#else
        throw std::runtime_error("CUDA support not compiled");
#endif
    }
}

void Tensor::free() {
    if (data_ && owns_data_) {
        if (device_ == DeviceType::CPU) {
            std::free(data_);
        } else if (device_ == DeviceType::CUDA) {
#ifdef MINISGL_USE_CUDA
            cudaFree(data_);
#endif
        }
    }
    data_ = nullptr;
    owns_data_ = false;
}

void Tensor::copy_from(const void* src, size_t size_bytes, DeviceType src_device) {
    if (!data_) return;
    if (device_ == DeviceType::CPU && src_device == DeviceType::CPU) {
        std::memcpy(data_, src, size_bytes);
    } else {
#ifdef MINISGL_USE_CUDA
        cudaMemcpyKind kind = cudaMemcpyHostToDevice;
        if (src_device == DeviceType::CPU && device_ == DeviceType::CUDA) kind = cudaMemcpyHostToDevice;
        else if (src_device == DeviceType::CUDA && device_ == DeviceType::CPU) kind = cudaMemcpyDeviceToHost;
        else if (src_device == DeviceType::CUDA && device_ == DeviceType::CUDA) kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy(data_, src, size_bytes, kind);
#else
        throw std::runtime_error("CUDA copy not supported in CPU mode");
#endif
    }
}

void Tensor::copy_to(void* dst, size_t size_bytes, DeviceType dst_device) const {
    if (!data_) return;
    if (device_ == DeviceType::CPU && dst_device == DeviceType::CPU) {
        std::memcpy(dst, data_, size_bytes);
    } else {
#ifdef MINISGL_USE_CUDA
        cudaMemcpyKind kind = cudaMemcpyDeviceToHost;
        if (device_ == DeviceType::CUDA && dst_device == DeviceType::CPU) kind = cudaMemcpyDeviceToHost;
        else if (device_ == DeviceType::CPU && dst_device == DeviceType::CUDA) kind = cudaMemcpyHostToDevice;
        else if (device_ == DeviceType::CUDA && dst_device == DeviceType::CUDA) kind = cudaMemcpyDeviceToDevice;
        cudaMemcpy(dst, data_, size_bytes, kind);
#else
        throw std::runtime_error("CUDA copy not supported in CPU mode");
#endif
    }
}

std::shared_ptr<Tensor> Tensor::to(DeviceType new_device) const {
    auto new_tensor = Tensor::create(shape_, dtype_, new_device);
    copy_to(new_tensor->data(), size_in_bytes(), new_device);
    return new_tensor;
}

} // namespace minisgl
