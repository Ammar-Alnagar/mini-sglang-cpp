#ifndef MINISGL_MODEL_QWEN3_H
#define MINISGL_MODEL_QWEN3_H

#include "minisgl/core/tensor.h"
#include "minisgl/layer/layer.h"
#include "minisgl/model/config.h"
#include "minisgl/model/weight.h"
#include <memory>
#include <vector>

namespace minisgl {

class Qwen3DecoderLayer {
  public:
    Qwen3DecoderLayer(const ModelConfig& config, int layer_id);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor>& x, std::shared_ptr<Tensor>& residual,
                                    const std::shared_ptr<Tensor>& positions);

    RMSNorm input_layernorm;
    Linear qkv_proj;
    Linear o_proj;
    RMSNorm post_attention_layernorm;
    Linear gate_up_proj;
    Linear down_proj;
    RotaryEmbedding rotary_emb;
    SwiGLU swiglu;

  private:
    int layer_id_;
    ModelConfig config_;
};

class Qwen3ForCausalLM {
  public:
    explicit Qwen3ForCausalLM(const ModelConfig& config);

    void load_weights(WeightLoader& loader);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& input_ids,
                                    const std::shared_ptr<Tensor>& positions);

    Embedding embed_tokens;
    std::vector<std::unique_ptr<Qwen3DecoderLayer>> layers;
    RMSNorm norm;
    Linear lm_head;

  private:
    ModelConfig config_;
};

} // namespace minisgl

#endif // MINISGL_MODEL_QWEN3_H
