# syntax=docker/dockerfile:1.7
# Multi-stage CI runner image.
ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04

# ── runtime ──
# devel (not runtime) base: nvcc is needed at runtime for tilelang JIT.
FROM ${BASE_IMAGE} AS runtime
# python-is-python3: later stages call `python`. PIP_BREAK_SYSTEM_PACKAGES: bypass PEP 668.
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip python-is-python3 git curl ca-certificates \
        build-essential cmake ninja-build unzip xz-utils sudo \
    && rm -rf /var/lib/apt/lists/*
COPY constraints.txt /tmp/constraints.txt
# `+cu129` pin forces the cu129 wheel from the pytorch index; the PyPI extra-index
# supplies torch's pure-python deps, which the pytorch index does not host.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/constraints.txt "torch==2.10.0+cu129"
RUN python -m pip install -c /tmp/constraints.txt \
        setuptools wheel ninja \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"

# Build tilelang once from the pinned commit; install --no-deps so pip never re-resolves it.
ARG TILELANG_GIT_SHA=65dbc9837beedf6882a40a08e18ea571d92fd6a5
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
# TORCH_CUDA_ARCH_LIST=9.0: the runner pool is Hopper (H200) only.
ENV MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} TORCH_CUDA_ARCH_LIST=9.0
RUN git clone https://github.com/tile-ai/tilelang.git /tmp/tilelang \
    && cd /tmp/tilelang \
    && git checkout ${TILELANG_GIT_SHA} \
    && git submodule update --init --recursive \
    && mkdir -p /opt/tilelang-wheels \
    && python -m pip wheel . --no-deps --no-build-isolation -w /opt/tilelang-wheels \
    && python -m pip install --no-deps /opt/tilelang-wheels/tilelang-*.whl \
    && rm -rf /tmp/tilelang
RUN python -c "import importlib.metadata as m, tilelang, torch, triton; \
print('tilelang', m.version('tilelang')); print('torch', torch.__version__); print('triton', triton.__version__)"

# ── post-fa3 ──
FROM runtime AS post-fa3
RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt \
        "pytest==9.0.2" "pytest-xdist>=3.0" "ruff==0.14.13"
# Comparison baselines are best-effort: a failed install must not break the image.
RUN python -m pip install --no-cache-dir --no-build-isolation "flash-attn-interface>=2.8.3" \
    || echo "WARNING: flash-attn-interface install failed (non-fatal)"

# ── fullstack ──
FROM post-fa3 AS fullstack
RUN python -m pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
        "flash-attn==2.8.3" \
        "flash-linear-attention==0.4.2" \
        "flashinfer>=0.6.6" \
        "vllm==0.18.0" \
        "sgl-kernel==0.3.21" \
    || echo "WARNING: a bench baseline install failed (non-fatal)"

# ── final ──
FROM fullstack AS final
# Cache dir defaults (host bind-mounts /ci-cache at runtime; pre-created below so they
# stay writable when run unmounted). PIP_NO_CACHE_DIR=0 re-enables caching (build stages disable it).
ENV AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool \
    TILELANG_CACHE_DIR=/ci-cache/tilelang \
    TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
    TRITON_CACHE_DIR=/ci-cache/triton \
    PIP_CACHE_DIR=/ci-cache/pip \
    PIP_NO_CACHE_DIR=0
ARG RUNNER_VERSION=2.334.0
# useradd before WORKDIR: home stays ci-runner-owned, and WORKDIR precedes the relative-path RUN (hadolint).
RUN useradd -m -s /bin/bash ci-runner
WORKDIR /home/ci-runner/runner
RUN mkdir -p /home/ci-runner/runner/_work/_tool \
    && curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && mkdir -p /ci-cache/pip /ci-cache/tilelang/tmp /ci-cache/triton \
    && chown -R ci-runner:ci-runner /home/ci-runner /ci-cache
# chmod as root before the USER drop.
COPY --chown=ci-runner:ci-runner .github/runner/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
USER ci-runner
ENTRYPOINT ["./entrypoint.sh"]
