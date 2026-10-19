#ifndef MINISGL_MODEL_QWEN3_MOE_H
#define MINISGL_MODEL_QWEN3_MOE_H

#include "minisgl/core/tensor.h"
#include "minisgl/layer/layer.h"
#include "minisgl/model/config.h"
#include "minisgl/model/weight.h"
#include <memory>
#include <vector>

namespace minisgl {

class Qwen3MoEDecoderLayer {
  public:
    Qwen3MoEDecoderLayer(const ModelConfig& config, int layer_id);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& residual,
                                    const std::shared_ptr<Tensor>& positions);

    RMSNorm input_layernorm;
    Linear qkv_proj;
    Linear o_proj;
    RMSNorm post_attention_layernorm;
    MoERouter router;
    std::vector<std::unique_ptr<Linear>> expert_gate_up;
    std::vector<std::unique_ptr<Linear>> expert_down;
    RotaryEmbedding rotary_emb;
    SwiGLU swiglu;

  private:
    int layer_id_;
    ModelConfig config_;
};

class Qwen3MoEForCausalLM {
  public:
    explicit Qwen3MoEForCausalLM(const ModelConfig& config);

    void load_weights(WeightLoader& loader);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids,
                                    const std::shared_ptr<Tensor>& positions);

    Embedding embed_tokens;
    std::vector<std::unique_ptr<Qwen3MoEDecoderLayer>> layers;
    RMSNorm norm;
    Linear lm_head;

  private:
    ModelConfig config_;
};

} // namespace minisgl

#endif // MINISGL_MODEL_QWEN3_MOE_H
