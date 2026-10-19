# System Architecture of Mini-SGLang Native Engine

## Architecture Overview

Mini-SGLang is constructed as a native C++/CUDA inference system composed of C++ runtime components:

```mermaid
flowchart TB
    subgraph Frontend [HTTP & CLI Layer]
        API[HttpServer - cpp-httplib]
        Shell[Interactive Terminal Shell]
    end

    subgraph Processing [Tokenizer & Preprocessing]
        Tokenizer[Native BPE Tokenizer]
    end

    subgraph Scheduling [Scheduler & KV Cache Management]
        Scheduler[Request Scheduler]
        RadixCache[Radix Cache Tree]
        KVPool[MHAKVCachePool]
        Scheduler <--> RadixCache
        RadixCache <--> KVPool
    end

    subgraph Execution [Model Execution & Compute Engine]
        ModelExec[Model Executor: Qwen3 / Qwen3-MoE]
        CUDAKernels[Custom CUDA C++ Kernels]
        TPComm[NCCL TPCommunicator]
        ModelExec --> CUDAKernels
        ModelExec --> TPComm
    end

    API --> Tokenizer
    Shell --> Tokenizer
    Tokenizer --> Scheduler
    Scheduler --> ModelExec
```

### Component Breakdown

1. **HttpServer**: Native C++ REST server (`cpp-httplib` and `nlohmann/json`) exposing OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/completions`, `/health`) and SSE streaming responses.
2. **Tokenizer**: Native C++ BPE tokenizer parsing Hugging Face `tokenizer.json` and `tokenizer_config.json`, applying Qwen chat templates, and performing incremental detokenization.
3. **Scheduler**: Manages request lifecycle (waiting queue -> prefill -> decode -> completion), page allocation in the KV cache, Radix Cache prefix matching, and chunked prefill chunking.
4. **Model Executor**: Native C++ model implementations for Qwen3 and Qwen3-MoE (`Qwen3ForCausalLM`, `Qwen3MoEForCausalLM`) with layer-by-layer forward execution.
5. **CUDA Kernels**: Custom CUDA C++ kernels for RMSNorm, Rotary Position Embeddings (RoPE), Index Gathering, and KV Page Store.
6. **TPCommunicator**: Native NCCL communicator performing tensor-parallel all-reduce and all-gather operations across GPU ranks.

---

## Detailed Systems Workflows & Diagrams

### 1. Request Lifecycle & Scheduler State Machine

```mermaid
stateDiagram-v2
    [*] --> Waiting: Request Submitted

    Waiting --> Prefill: Scheduler Selects Request
    note right of Prefill
      Matches Radix Cache Tree.
      Allocates KV pages.
      Splits long prompts into chunks.
    end note

    Prefill --> ChunkedPrefill: Token Length > Chunk Limit
    ChunkedPrefill --> Prefill: Process Next Chunk
    Prefill --> Decode: All Prefill Tokens Processed

    state Decode {
      [*] --> ForwardPass
      ForwardPass --> SampleToken
      SampleToken --> UpdateKVCache
    }

    Decode --> Finished: EOS Token or Max Tokens Reached
    Finished --> [*]: Release KV Cache & Return Response
```

---

### 2. Radix Cache Page Reuse Engine

The Radix Cache maintains a tree structure of prefix token sequences. When a new request arrives, the scheduler traverses the Radix Cache tree to find the longest matching cached token prefix.

```mermaid
graph TD
    Root(("Root Node (System Prompt)"))

    Root --> NodeA["Token Prefix A: [ 'User:', 'Hello' ]\n(Pages 1, 2 Cached)"]
    Root --> NodeB["Token Prefix B: [ 'User:', 'System Info' ]\n(Pages 3, 4 Cached)"]

    NodeA --> ChildA1["Request 1 Branch: [ 'Explain C++' ]\n(Page 5)"]
    NodeA --> ChildA2["Request 2 Branch: [ 'Explain CUDA' ]\n(Page 6)"]

    style Root fill:#f9f,stroke:#333,stroke-width:2px
    style NodeA fill:#bbf,stroke:#333,stroke-width:2px
    style NodeB fill:#bbf,stroke:#333,stroke-width:2px
```

When Request 2 arrives sharing `[ 'User:', 'Hello' ]`, pages 1 and 2 are reused instantly, avoiding redundant KV computation.

---

### 3. Chunked Prefill Execution Pipeline

Chunked prefill prevents peak GPU memory spikes caused by long input prompts by partitioning the prefill stage into fixed-size chunks (`--max-prefill-length`).

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant E as Model Executor
    participant KV as KV Cache Pool

    Note over S: Input Prompt: 4096 Tokens<br/>Chunk Size: 1024 Tokens

    S->>KV: Allocate Page Table for Request

    loop Chunk 1 to 4
        S->>E: Execute Forward Pass for Chunk i (Tokens [i*1024 : (i+1)*1024])
        E->>KV: Write Key-Value States to Paged Memory
        Note over E: Intermediate Decode Requests Batched Simultaneously
    end

    S->>S: Transition Request to Decode Queue
```

---

### 4. Distributed Tensor Parallelism (NCCL) Architecture

Tensor Parallelism distributes linear projections across multiple GPU devices using NCCL collectives.

```mermaid
flowchart LR
    subgraph Rank 0 [GPU Rank 0]
        W1_0[Weight Partition 0]
        Q1[Q/K/V Projection]
        Attn0[Attention Compute]
    end

    subgraph Rank 1 [GPU Rank 1]
        W1_1[Weight Partition 1]
        Q2[Q/K/V Projection]
        Attn1[Attention Compute]
    end

    Input[Input Tensor] --> W1_0
    Input --> W1_1

    Q1 --> Attn0
    Q2 --> Attn1

    Attn0 --> NCCL{NCCL All-Reduce / All-Gather}
    Attn1 --> NCCL

    NCCL --> Output[Combined Layer Output]
```

---

## Directory Structure

- `include/minisgl/core`: Tensor memory abstractions, data types, requests, batching, and context.
- `include/minisgl/model`: Model configurations (`ModelConfig`), Safetensors `WeightLoader`, Qwen3 and Qwen3-MoE model headers.
- `include/minisgl/layer`: Neural network layers (Embedding, RMSNorm, Linear, RotaryEmbedding, SwiGLU, MoERouter).
- `include/minisgl/cache`: Paged KV cache allocation (`MHAKVCachePool`), Radix Cache, and Naive Cache.
- `include/minisgl/scheduler`: Request scheduler and batching engine.
- `include/minisgl/distributed`: Tensor Parallelism communicator (`TPCommunicator`).
- `include/minisgl/tokenizer`: Native Tokenizer and Detokenizer.
- `include/minisgl/server`: HTTP API Server, CLI argument parser, and Interactive Terminal Shell.
- `kernels/cuda`: Custom CUDA C++ kernels (`rmsnorm.cu`, `rotary.cu`, `index.cu`, `radix.cu`).
- `src/`: Native C++ source implementations.
- `tests/`: CTest unit, CUDA, and integration test suites.
- `benchmark/`: Benchmark executables (`bench_offline`, `bench_online`).
- `docker/`: C++ multi-stage Dockerfile.
