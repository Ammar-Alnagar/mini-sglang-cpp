#ifndef MINISGL_MODEL_WEIGHT_H
#define MINISGL_MODEL_WEIGHT_H

#include "minisgl/core/tensor.h"
#include "minisgl/model/config.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace minisgl {

struct TensorInfo {
    std::string name;
    DataType dtype;
    std::vector<int64_t> shape;
    size_t data_offset_start;
    size_t data_offset_end;
    std::string file_path;
};

class WeightLoader {
  public:
    WeightLoader(std::string model_dir);

    void load_metadata();
    std::shared_ptr<Tensor> load_tensor(const std::string& tensor_name,
                                        DeviceType device = DeviceType::CPU);
    bool has_tensor(const std::string& tensor_name) const;

    const std::unordered_map<std::string, TensorInfo>& tensor_infos() const {
        return tensor_infos_;
    }

  private:
    std::string model_dir_;
    std::unordered_map<std::string, TensorInfo> tensor_infos_;

    void load_safetensors_file(const std::string& filepath);
};

} // namespace minisgl

#endif // MINISGL_MODEL_WEIGHT_H
