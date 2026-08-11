# syntax=docker/dockerfile:1.7
# Multi-stage CI runner image for the self-hosted GPU runner.
# cu132 is the newest variant the PyTorch index publishes; the base must match the torch wheels.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu22.04

# ── runtime ──
# devel, not runtime: nvcc is needed at build (FA3, tilelang) and at runtime (tilelang JIT).
FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
# 22.04 ships python3.10; 3.12 comes from deadsnakes, whose PPA tooling is purged after use.
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
COPY constraints.txt constraints-runner-lock.txt /tmp/
# Env, not a flag per install: it also reaches build backends under --no-build-isolation, and a
# step added later cannot forget it. PIP_CONSTRAINT still matters — `pip wheel` has no uv
# equivalent, so the tilelang stage builds through pip.
ENV PIP_CONSTRAINT="/tmp/constraints.txt /tmp/constraints-runner-lock.txt"
# uv resolves and installs everything below; pip only bootstraps it. constraints.txt pins its
# version, since the installer cannot come from the lock it produces.
RUN python -m pip install uv
# These mirror the header of constraints-runner-lock.txt: uv defaults to first-index, which would
# resolve a package to whichever index lists it first rather than to the version the lock was
# compiled against. Install and compile must agree on all four or the lock stops applying.
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu132 \
    UV_EXTRA_INDEX_URL="https://pypi.org/simple https://download.pytorch.org/whl/cpu" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_CONSTRAINT="/tmp/constraints.txt /tmp/constraints-runner-lock.txt" \
    UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_NO_CACHE=1
# cu132 wheels from the pytorch index; PyPI's default is another variant. The CPU index is here
# for torchaudio alone: no cu132 build of it exists, and a CUDA build of a different minor refuses
# to load next to this torch, so vllm's pin is met by the extension-free CPU wheel.
RUN uv pip install "torch==2.13.0+cu132" "torchvision==0.28.0+cu132"
# tilelang's build backend and runtime deps, so its stage can build with --no-build-isolation.
# The wheel cmake meets tilelang's >=3.26.1 floor; jammy apt cmake is 3.22.
RUN uv pip install \
        setuptools wheel ninja scikit-build-core patchelf cmake \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
# ARCH_LIST 9.0: the pool is Hopper (H200) only. CUDA_HOME/PATH pin the base toolkit — bench
# baselines drop a later nvcc into site-packages, and building against that mismatches the runtime.
ENV MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} TORCH_CUDA_ARCH_LIST=9.0 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH}

# ── post-fa3 ──
# Test tooling + FlashAttention-3 (a bench baseline). tilelang is built last; see its stage.
FROM runtime AS post-fa3
RUN uv pip install pytest pytest-xdist ruff pytest-timeout py-spy
# No PyPI wheel for Hopper FA3 — build from hopper/. A commit, not the v2.8.3 tag: releases up to
# 2.8.3.post1 swap in a downloaded CUDA 12.6 nvcc whenever the system CUDA is not exactly 12.8,
# and that standalone toolchain ships no CCCL, so every translation unit fails on
# `cuda/std/utility`. Only main takes the CUDA 13 path: system nvcc, plus the CCCL headers CUDA 13
# moved to include/cccl. Only csrc/cutlass: --recursive also pulls AMD composable_kernel, unused
# at sm_90 and slow to clone. Non-fatal; the tree is always removed.
ARG FLASH_ATTENTION_GIT_SHA=a369df707e1980fb328abcc1733e3457ec10155f
RUN { mkdir -p /tmp/flash-attention \
        && cd /tmp/flash-attention \
        && git init -q \
        && git remote add origin https://github.com/Dao-AILab/flash-attention.git \
        && git fetch --depth 1 origin "${FLASH_ATTENTION_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && git submodule update --init --depth 1 csrc/cutlass \
        && cd hopper && python setup.py install; }; \
    status=$?; rm -rf /tmp/flash-attention; \
    [ "$status" -eq 0 ] || echo "WARNING: FlashAttention-3 (hopper) build failed (non-fatal)"

# ── fa2 ──
# Own layer: no wheel for this torch/py, so it builds from source (~minutes) and changes to the
# bench loop below must not recompile it. Non-fatal.
FROM post-fa3 AS fa2
RUN uv pip install --no-build-isolation "flash-attn==2.8.3.post1" \
    || echo "WARNING: flash-attn (FA2) install failed (non-fatal)"

# ── fullstack ──
# Remaining bench baselines, non-fatal. This vllm and no other: its apache-tvm-ffi pin (0.1.11)
# is the one the tilelang commit needs. sgl-kernel is not installed.
FROM fa2 AS fullstack
RUN for pkg in \
        "flash-linear-attention==0.5.2" \
        "vllm==0.27.1"; do \
        uv pip install --no-build-isolation "$pkg" \
            || echo "WARNING: bench baseline '$pkg' install failed (non-fatal)"; \
    done
# mamba_ssm's Triton kernels, a Mamba bench baseline. --no-deps so it cannot pull its own newer
# tilelang pin; the CUDA extension must still build because importing it loads selective_scan_cuda.
RUN MAMBA_FORCE_BUILD=TRUE \
    uv pip install --no-build-isolation --no-deps "mamba-ssm==2.3.2.post1" \
    || echo "WARNING: mamba-ssm 2.3.2.post1 install failed (non-fatal)"
# DeepGEMM, a grouped-GEMM bench baseline, from the official tag (PyPI's 1.0.0 sdist is stale).
ARG DEEPGEMM_GIT_SHA=c9f8b34dcdacc20aa746b786f983492c51072870
RUN { mkdir -p /tmp/DeepGEMM \
        && cd /tmp/DeepGEMM \
        && git init -q \
        && git remote add origin https://github.com/deepseek-ai/DeepGEMM.git \
        && for attempt in 1 2 3 4 5; do \
            git -c http.version=HTTP/1.1 fetch --depth 1 origin "${DEEPGEMM_GIT_SHA}" && break; \
            status="$?"; \
            echo "WARNING: DeepGEMM fetch failed (attempt ${attempt}/5, status ${status})"; \
            if [ "${attempt}" = 5 ]; then exit "${status}"; fi; \
            sleep "$((attempt * 10))"; \
        done \
        && git checkout -q FETCH_HEAD \
        && for attempt in 1 2 3 4 5; do \
            git -c http.version=HTTP/1.1 submodule update --init --depth 1 --jobs 1 \
                third-party/cutlass third-party/fmt && break; \
            status="$?"; \
            echo "WARNING: DeepGEMM submodule fetch failed (attempt ${attempt}/5, status ${status})"; \
            rm -rf third-party/cutlass third-party/fmt .git/modules/third-party/cutlass .git/modules/third-party/fmt; \
            if [ "${attempt}" = 5 ]; then exit "${status}"; fi; \
            sleep "$((attempt * 10))"; \
        done \
        && DG_FORCE_BUILD=1 uv pip install --no-build-isolation --no-deps .; } \
    || echo "WARNING: DeepGEMM v2.1.1.post3 install failed (non-fatal)"; \
    rm -rf /tmp/DeepGEMM

# ── tilelang ──
# Built LAST: a SHA bump rebuilds only this layer, and tilelang links against the final stack.
# The ARGs live here so changing them never invalidates the bench layers above. vllm pins an exact
# tilelang release, which this stage overwrites — leaving that pin unsatisfied is the point.
FROM fullstack AS tilelang
ARG TILELANG_GIT_SHA
ARG TILELANG_VERSION
# main mode:    --build-arg TILELANG_GIT_SHA=<commit>  → clone + compile that commit
# release mode: --build-arg TILELANG_VERSION=<version> → pip install the PyPI release
# --no-deps: never re-resolve the cu132 stack. --reinstall: a same-version PyPI tilelang must not
# read as already satisfied. `pip wheel` stays on pip — uv has no equivalent subcommand.
RUN if [ -n "${TILELANG_GIT_SHA}" ]; then \
        mkdir -p /tmp/tilelang && cd /tmp/tilelang \
        && git init -q \
        && git remote add origin https://github.com/tile-ai/tilelang.git \
        && git fetch --depth 1 origin "${TILELANG_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && git submodule update --init --depth 1 --recursive \
            3rdparty/composable_kernel 3rdparty/cutlass 3rdparty/tvm \
        && mkdir -p /opt/tilelang-wheels \
        && CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
            python -m pip wheel . --no-deps --no-build-isolation -w /opt/tilelang-wheels \
        && uv pip install --no-deps --reinstall /opt/tilelang-wheels/tilelang-*.whl \
        && rm -rf /tmp/tilelang; \
    elif [ -n "${TILELANG_VERSION}" ]; then \
        uv pip install --no-deps --reinstall "tilelang==${TILELANG_VERSION}"; \
    else \
        echo "ERROR: set --build-arg TILELANG_GIT_SHA=<commit> (main) or TILELANG_VERSION=<ver> (release)"; \
        exit 1; \
    fi
# CUPTI binding for benchmark timing. --no-deps: it must not re-resolve a CUDA wheel under torch.
RUN uv pip install --no-deps "cupti-python==13.2.0"
# Fail the build on an incoherent stack. GPU-free; the script's docstring lists what it checks.
COPY scripts/ci/verify_runtime_stack.py /tmp/verify_runtime_stack.py
RUN python /tmp/verify_runtime_stack.py

# ── final ──
FROM tilelang AS final
# Cache defaults. The host bind-mounts /ci-cache; the dirs are pre-created for unmounted runs.
ENV AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool \
    TILELANG_CACHE_DIR=/ci-cache/tilelang \
    TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
    TRITON_CACHE_DIR=/ci-cache/triton \
    PIP_CACHE_DIR=/ci-cache/pip \
    PIP_NO_CACHE_DIR=0
ARG RUNNER_VERSION=2.336.0
# UID/GID 1000: the account must own the shared /ci-cache bind-mount across hosts. The chown below
# matters only unmounted — in CI the bind-mount shadows those dirs. useradd before WORKDIR (hadolint).
RUN groupadd -g 1000 ci-runner && useradd -u 1000 -g 1000 -m -s /bin/bash ci-runner
WORKDIR /home/ci-runner/runner
RUN mkdir -p /home/ci-runner/runner/_work/_tool \
    && curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /ci-cache/pip /ci-cache/tilelang/tmp /ci-cache/triton \
    && chown -R ci-runner:ci-runner /home/ci-runner /ci-cache
# chmod as root before the USER drop.
COPY --chown=ci-runner:ci-runner .github/runner/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
USER ci-runner
ENTRYPOINT ["./entrypoint.sh"]
