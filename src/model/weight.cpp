#include "minisgl/model/weight.h"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <nlohmann/json.hpp>

namespace minisgl {

namespace fs = std::filesystem;

static DataType string_to_dtype(const std::string& str) {
    if (str == "F32") return DataType::Float32;
    if (str == "F16") return DataType::Float16;
    if (str == "BF16") return DataType::BFloat16;
    if (str == "I32") return DataType::Int32;
    if (str == "I64") return DataType::Int64;
    if (str == "I8") return DataType::Int8;
    return DataType::Float32;
}

WeightLoader::WeightLoader(std::string model_dir) : model_dir_(std::move(model_dir)) {
    load_metadata();
}

void WeightLoader::load_safetensors_file(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return;

    uint64_t header_size = 0;
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));

    std::vector<char> header_buf(header_size);
    file.read(header_buf.data(), header_size);

    nlohmann::json header = nlohmann::json::parse(header_buf.begin(), header_buf.end());

    size_t buffer_base_offset = 8 + header_size;

    for (auto& [key, value] : header.items()) {
        if (key == "__metadata__") continue;

        TensorInfo info;
        info.name = key;
        info.file_path = filepath;
        info.dtype = string_to_dtype(value["dtype"].get<std::string>());
        info.shape = value["shape"].get<std::vector<int64_t>>();

        auto offsets = value["data_offsets"].get<std::vector<size_t>>();
        info.data_offset_start = buffer_base_offset + offsets[0];
        info.data_offset_end = buffer_base_offset + offsets[1];

        tensor_infos_[key] = info;
    }
}

void WeightLoader::load_metadata() {
    tensor_infos_.clear();
    if (!fs::exists(model_dir_)) return;

    for (const auto& entry : fs::directory_iterator(model_dir_)) {
        if (entry.path().extension() == ".safetensors") {
            load_safetensors_file(entry.path().string());
        }
    }
}

bool WeightLoader::has_tensor(const std::string& tensor_name) const {
    return tensor_infos_.find(tensor_name) != tensor_infos_.end();
}

std::shared_ptr<Tensor> WeightLoader::load_tensor(const std::string& tensor_name, DeviceType device) {
    auto it = tensor_infos_.find(tensor_name);
    if (it == tensor_infos_.end()) {
        throw std::runtime_error("Tensor not found in weights: " + tensor_name);
    }

    const auto& info = it->second;
    auto tensor = Tensor::create(info.shape, info.dtype, DeviceType::CPU);

    std::ifstream file(info.file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open safetensors file: " + info.file_path);
    }

    file.seekg(info.data_offset_start);
    size_t bytes_to_read = info.data_offset_end - info.data_offset_start;
    file.read(reinterpret_cast<char*>(tensor->data()), bytes_to_read);

    if (device == DeviceType::CUDA) {
        return tensor->to(DeviceType::CUDA);
    }

    return tensor;
}

} // namespace minisgl
