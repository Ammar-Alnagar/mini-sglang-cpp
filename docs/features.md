# Features of Mini-SGLang Native Engine

## Online Serving

Mini-SGLang provides an OpenAI-compatible HTTP API server written in C++. Supported endpoints include:

- `/v1/chat/completions`: Chat completions supporting streaming (Server-Sent Events) and non-streaming responses.
- `/v1/completions`: Text completions endpoint.
- `/health`: Service health check returning HTTP 200.

## Interactive Shell Mode

An interactive terminal shell is available for prompt testing and direct conversation. The shell maintains chat history context across turns. Enter `/reset` to reset session history.

Example:

```bash
./build/minisgl --model Qwen/Qwen3-0.6B --shell
```

## Distributed Serving

Tensor Parallelism (TP) scales inference across multiple NVIDIA GPUs using native NCCL collective communication (`ncclAllReduce`, `ncclAllGather`). Enable multi-GPU execution using the `--tp N` flag.

## Supported Models

Mini-SGLang natively supports:

- Qwen3 dense model architectures.
- Qwen3-MoE Mixture-of-Experts model architectures with top-k expert routing.

## Chunked Prefill

Chunked prefill splits long prompt token sequences into configurable chunks during prefill execution. This limits peak KV cache memory usage during prefill. Configure maximum prefill chunk size with `--max-prefill-length N`.

## Radix Cache

Mini-SGLang manages Key-Value (KV) cache pages with a Radix Cache tree structure. Token sequences are matched against existing cached prefixes to reuse previously computed KV states across requests. Switch to naive page allocation using `--cache naive`.

## Overlap Scheduling

CPU host request scheduling is overlapped with GPU kernel execution across CUDA streams to reduce scheduling overhead and maximize GPU compute saturation.
