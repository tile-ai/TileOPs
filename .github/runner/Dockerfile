# syntax=docker/dockerfile:1.7
# Multi-stage CI runner image.
ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04

# ── runtime ──
# devel (not runtime) base: nvcc is needed at runtime for tilelang JIT.
FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
# 22.04 ships python3.10; install 3.12 from deadsnakes and make it the default python.
# software-properties-common is purged after adding the PPA to keep the image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git build-essential cmake ninja-build unzip xz-utils sudo \
    && apt-get purge -y --auto-remove software-properties-common \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m ensurepip --upgrade \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && python -m pip install --upgrade pip
COPY constraints.txt /tmp/constraints.txt
# `+cu129` pin forces the cu129 wheel from the pytorch index; the PyPI extra-index
# supplies torch's pure-python deps, which the pytorch index does not host.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/constraints.txt "torch==2.10.0+cu129"
# scikit-build-core + patchelf are tilelang's PEP 517 build backend (its
# [build-system].requires); --no-build-isolation in the next stage means pip will NOT
# auto-install them, so they must be present here. cmake from pip provides >=3.26.1
# (tilelang's [tool.scikit-build] floor); the jammy apt cmake is 3.22, too old.
RUN python -m pip install -c /tmp/constraints.txt \
        setuptools wheel ninja scikit-build-core patchelf cmake \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"

# Build tilelang once from the pinned commit; install --no-deps so pip never re-resolves it.
# TILELANG_GIT_SHA is a required build-arg (no default): bump it via `--build-arg` and tag
# the image `:<short-sha>` — the Dockerfile never carries a SHA literal.
ARG TILELANG_GIT_SHA
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
# TORCH_CUDA_ARCH_LIST=9.0: the runner pool is Hopper (H200) only.
ENV MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} TORCH_CUDA_ARCH_LIST=9.0
RUN if [ -z "${TILELANG_GIT_SHA}" ]; then \
        echo "ERROR: build with --build-arg TILELANG_GIT_SHA=<commit>"; exit 1; \
    fi \
    && git clone https://github.com/tile-ai/tilelang.git /tmp/tilelang \
    && cd /tmp/tilelang \
    && git checkout "${TILELANG_GIT_SHA}" \
    && git submodule update --init --recursive \
    && mkdir -p /opt/tilelang-wheels \
    && python -m pip wheel . --no-deps --no-build-isolation -w /opt/tilelang-wheels \
    && python -m pip install --no-deps /opt/tilelang-wheels/tilelang-*.whl \
    && rm -rf /tmp/tilelang
RUN python -c "import importlib.metadata as m, tilelang, torch, triton; \
print('tilelang', m.version('tilelang')); print('torch', torch.__version__); print('triton', triton.__version__)"

# ── post-fa3 ──
# Bench / test layer: a future slim image (RFC §9) stops at `runtime` and skips everything
# below. pytest/ruff are dev/test tooling; FlashAttention-3 below is a bench baseline.
FROM runtime AS post-fa3
RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt \
        "pytest==9.0.2" "pytest-xdist>=3.0" "ruff==0.14.13"
# FlashAttention-3 (Hopper) has no PyPI wheel — build it from the repo's hopper/ dir
# (--recursive pulls the csrc/cutlass submodule the build needs). Best-effort: a bench
# baseline must not break the image. The source tree is removed whether or not the build
# succeeds, so it never lands in the image.
RUN git clone --depth 1 --branch v2.8.3 --recursive \
        https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
        && cd /tmp/flash-attention/hopper && python setup.py install; \
    status=$?; rm -rf /tmp/flash-attention; \
    [ "$status" -eq 0 ] || echo "WARNING: FlashAttention-3 (hopper) build failed (non-fatal)"

# ── fullstack ──
# Bench baselines (comparison targets), dropped wholesale by a future slim image (RFC §9).
# Install each independently and best-effort: one missing or unbuildable package must not
# abort the rest or fail the image (flashinfer's PyPI name is `flashinfer-python`).
FROM post-fa3 AS fullstack
RUN for pkg in \
        "flash-attn==2.8.3" \
        "flash-linear-attention==0.4.2" \
        "flashinfer-python>=0.6.6" \
        "vllm==0.18.0" \
        "sgl-kernel==0.3.21"; do \
        python -m pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt "$pkg" \
            || echo "WARNING: bench baseline '$pkg' install failed (non-fatal)"; \
    done

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
