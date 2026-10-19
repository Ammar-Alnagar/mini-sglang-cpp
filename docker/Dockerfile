# Multi-stage Dockerfile for Mini-SGLang Native C++/CUDA Inference Engine

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON && \
    cmake --build build --config Release -j$(nproc)

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/minisgl /usr/local/bin/minisgl
COPY --from=builder /app/build/libminisgl_lib.a /usr/local/lib/libminisgl.a

EXPOSE 1919

ENTRYPOINT ["minisgl"]
CMD ["--model", "Qwen/Qwen3-0.6B", "--host", "0.0.0.0", "--port", "1919"]
