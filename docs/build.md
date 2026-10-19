# Build Instructions and Installation Guide

This document provides complete instructions for building and installing Mini-SGLang on Linux platforms, including prerequisites, environment setup, build configurations, and Docker execution.

---

## One-Line Quick Start Commands

### Standard CUDA Build (GPU Accelerated)

```bash
git clone https://github.com/sgl-project/mini-sglang.git && cd mini-sglang && ./build.sh
```

### CPU-Only Test Mode (No GPU required)

```bash
git clone https://github.com/sgl-project/mini-sglang.git && cd mini-sglang && ./build.sh --cpu
```

---

## Prerequisites

### Hardware Requirements

- **GPU Mode**: NVIDIA GPU with Compute Capability 8.0+ (Ampere, Hopper, Ada Lovelace, or newer).
- **CPU Mode**: Any x86_64 or aarch64 CPU supporting C++20.

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+, or Arch Linux).
- **C++ Compiler**: GCC 11+ or Clang 13+ with C++20 support.
- **CMake**: Version 3.22 or higher.
- **CUDA Toolkit** (Optional for CPU mode): NVIDIA CUDA Toolkit 12.0+ with `nvcc`.
- **NCCL** (Optional for Tensor Parallelism): NVIDIA Collective Communications Library 2.12+.

---

## Building with `./build.sh`

Mini-SGLang includes a shell script `./build.sh` that wraps CMake invocation:

```bash
./build.sh [options]
```

### Supported `./build.sh` Options

| Flag | Description |
| --- | --- |
| `--cuda`, `-g` | Enable CUDA compilation (`-DENABLE_CUDA=ON`) [Default] |
| `--cpu`, `-c` | Disable CUDA and build CPU test mode (`-DENABLE_CUDA=OFF`) |
| `--release` | Build in Release configuration (`-DCMAKE_BUILD_TYPE=Release`) [Default] |
| `--debug` | Build in Debug configuration (`-DCMAKE_BUILD_TYPE=Debug`) |
| `--clean` | Remove the `build/` directory before configuring |
| `--jobs N`, `-j N` | Specify the number of parallel compilation jobs |
| `--no-tests` | Disable unit test binaries compilation |
| `--no-benchmarks` | Disable benchmark binaries compilation |
| `--help`, `-h` | Display help message |

### Examples

```bash
# Build Debug mode with 8 parallel jobs
./build.sh --debug -j 8

# Clean re-build in CPU mode
./build.sh --clean --cpu

# Release build without benchmarks
./build.sh --no-benchmarks
```

---

## Manual CMake Build Workflow

If you prefer using CMake directly, follow these steps:

### 1. Clone Repository

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang
```

### 2. Configure CMake Build

```bash
# Release build with CUDA support
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# Alternatively, configure CPU test mode
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
```

### CMake Configuration Parameters

| CMake Option | Default | Description |
| --- | --- | --- |
| `CMAKE_BUILD_TYPE` | `Release` | Build mode (`Release`, `Debug`, `RelWithDebInfo`) |
| `ENABLE_CUDA` | `ON` | Enable CUDA C++ kernel compilation |
| `BUILD_TESTS` | `ON` | Build CTest unit and integration tests |
| `BUILD_BENCHMARKS` | `ON` | Build performance benchmarking executables |

### 3. Compile Project

```bash
cmake --build build --config Release -j$(nproc)
```

Upon successful compilation, the main executable `minisgl` and test binaries will be placed in the `build/` directory.

---

## Docker Build Workflow

Mini-SGLang provides a multi-stage Dockerfile located at `docker/Dockerfile`.

### Building Docker Image

```bash
docker build -t minisgl:latest -f docker/Dockerfile .
```

### Running Container with NVIDIA GPU Acceleration

```bash
docker run --gpus all \
  -p 1919:1919 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  minisgl:latest --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 1919
```

---

## Troubleshooting & Verification

### Verify Compilation

Run unit tests using CTest:

```bash
ctest --test-dir build --output-on-failure
```

### Common Issues

1. **CUDA Compiler Not Found**:
   Ensure `nvcc` is in your system `PATH` and CUDA libraries are in `LD_LIBRARY_PATH`.
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

2. **Out of Memory During Parallel Build**:
   Reduce parallel compilation jobs using `-j`:
   ```bash
   ./build.sh -j 2
   ```

3. **Missing CMake Dependencies**:
   Third-party C++ libraries (`nlohmann_json` and `cpp-httplib`) are fetched automatically by CMake FetchContent. Ensure your machine has internet access during initial configuration.
