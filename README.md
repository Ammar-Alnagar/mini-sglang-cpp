# Mini-SGLang

A lightweight, high-performance native C++/CUDA inference engine for Large Language Models.

---

Mini-SGLang is a native C++/CUDA LLM serving framework designed for low latency, high throughput, and memory efficiency. Built with native C++ systems code and custom CUDA C++ kernels, it provides a standalone, production-oriented LLM inference runtime with zero Python dependencies.

## Key Features

- High Performance: Native C++ runtime with optimized CUDA kernels for peak GPU utilization and minimal CPU scheduling overhead.
- Zero Python Dependencies: Entire runtime, tokenizer, model execution, KV cache, scheduler, and HTTP server are written in C++ and CUDA.
- Radix Cache: Reuses Key-Value (KV) cache across requests sharing common prefix tokens to eliminate duplicate prefill computation.
- Chunked Prefill: Splits long prompt prefill sequences into configurable chunks to manage peak GPU memory usage and avoid Out-Of-Memory errors.
- Overlap Scheduling: Overlaps CPU host scheduling preparation with GPU kernel execution across CUDA streams.
- Tensor Parallelism: Scales model inference across multiple GPUs using NCCL collective communication.
- Model Support: Native implementations for Qwen3 and Qwen3-MoE model architectures.

## Quick Start

### Prerequisites

- Linux (x86_64 or aarch64)
- C++20 compatible compiler (GCC 11+ or Clang 13+)
- CMake 3.22+
- NVIDIA CUDA Toolkit 12.0+ (optional for CUDA acceleration; CPU fallback available for testing)

### Building from Source

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# Configure CMake build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# Compile the project
cmake --build build --config Release -j$(nproc)
```

### Running the Server

Launch an OpenAI-compatible API server using the native executable:

```bash
# Serve Qwen/Qwen3-0.6B model on default port 1919
./build/minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 1919

# Serve with Tensor Parallelism across 4 GPUs
./build/minisgl --model Qwen/Qwen3-0.6B --tp 4 --port 1919
```

Once running, send inference requests using curl or standard OpenAI API clients:

```bash
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

### Interactive Shell Mode

Chat directly in the terminal using the interactive shell flag:

```bash
./build/minisgl --model Qwen/Qwen3-0.6B --shell
```

Use `/reset` in the shell to clear conversation history.

### Running with Docker

Build and run using the multi-stage Docker image:

```bash
# Build Docker image
docker build -t minisgl -f docker/Dockerfile .

# Launch server
docker run --gpus all -p 1919:1919 minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
```

## Running Benchmarks and Tests

Run the test suite using CTest:

```bash
ctest --test-dir build --output-on-failure
```

Run offline throughput benchmark:

```bash
./build/benchmark/bench_offline
```

Run online API latency benchmark:

```bash
./build/benchmark/bench_online
```

## Documentation

- Feature Details: See docs/features.md
- System Architecture: See docs/structures.md
- Feature Matrix: See docs/porting/feature-matrix.md
- Native Architecture: See docs/porting/native-architecture.md
