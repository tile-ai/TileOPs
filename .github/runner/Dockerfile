# syntax=docker/dockerfile:1.7
# cu132 is the newest variant the PyTorch index publishes; the base must match the torch wheels.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu22.04

# devel, not runtime: nvcc is needed at build (FA3, tilelang) and at runtime (tilelang JIT).
FROM ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
# 22.04 ships python3.10; 3.12 comes from deadsnakes.
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
COPY constraints.txt constraints-runner-lock.txt .github/runner/overrides.txt /tmp/
# Env, not a flag per install: it reaches build backends too, and a step added later cannot forget
# it. pip keeps its own copy because the tilelang stage builds the wheel through `pip wheel`.
ENV PIP_CONSTRAINT="/tmp/constraints.txt /tmp/constraints-runner-lock.txt"
# pip only bootstraps uv; constraints.txt pins it, since the installer cannot come from the lock
# it produces.
RUN python -m pip install uv
# These mirror the lock's header; UV_OVERRIDE is its --override. uv defaults to first-index, which
# resolves a package against whichever index lists it first rather than the one the lock was
# compiled against; install and compile must agree on all of them or the lock stops applying.
ENV UV_INDEX_URL=https://download.pytorch.org/whl/cu132 \
    UV_EXTRA_INDEX_URL="https://pypi.org/simple https://download.pytorch.org/whl/cpu" \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_CONSTRAINT="/tmp/constraints.txt /tmp/constraints-runner-lock.txt" \
    UV_OVERRIDE=/tmp/overrides.txt \
    UV_SYSTEM_PYTHON=1 UV_BREAK_SYSTEM_PACKAGES=1 UV_NO_CACHE=1
# The CPU index serves torchaudio alone: no cu132 build of it exists, and a CUDA build of another
# minor refuses to load beside this torch, so vllm's pin is met by the CPU wheel, whose one
# extension targets torch's stable ABI and carries no CUDA.
RUN uv pip install "torch==2.14.0+cu132" "torchvision==0.29.0+cu132"
# tilelang's build backend and runtime deps, so its stage can use --no-build-isolation. The wheel
# cmake meets tilelang's >=3.26.1 floor; jammy apt cmake is 3.22. setuptools-scm is tileops'
# backend, baked in for the same reason.
RUN uv pip install \
        setuptools setuptools-scm wheel ninja scikit-build-core patchelf cmake \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
# ARCH_LIST 9.0: the pool is Hopper (H200) only. CUDA_HOME/PATH pin the base toolkit — bench
# baselines drop a later nvcc into site-packages, and building against that mismatches the runtime.
ENV MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} TORCH_CUDA_ARCH_LIST=9.0 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH}

FROM runtime AS post-fa3
RUN uv pip install pytest pytest-xdist ruff pytest-timeout pytest-cov py-spy
# Hopper FA3 has no wheel. A commit, not the v2.8.3 tag: releases through 2.8.3.post1 swap in a
# downloaded CUDA 12.6 nvcc unless the system CUDA is exactly 12.8, and that toolchain ships no
# CCCL, so every translation unit fails on `cuda/std/utility`. Only main takes the CUDA 13 path.
# FA2 builds from the same commit, which compiles its extension as C++20; torch 2.14's headers
# reject the C++17 every 2.8.3 release passes.
# Only csrc/cutlass: --recursive also pulls AMD composable_kernel, unused at sm_90 and slow.
ARG FLASH_ATTENTION_GIT_SHA=616b0e8abab13b87b01525b3916d5a863ab02ae0
# Fatal, unlike the bench baselines below: eleven attention benchmarks name `fa3` and
# lose the column without it. Imported here because a returned 0 has meant an installed
# wrapper with no compiled extension behind it.
RUN set -e; \
    mkdir -p /tmp/flash-attention \
        && cd /tmp/flash-attention \
        && git init -q \
        && git remote add origin https://github.com/Dao-AILab/flash-attention.git \
        && git fetch --depth 1 origin "${FLASH_ATTENTION_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && git submodule update --init --depth 1 csrc/cutlass; \
    cd /tmp/flash-attention/hopper && python setup.py install; \
    cd / && rm -rf /tmp/flash-attention; \
    python -c "import flash_attn_interface as m; assert m.flash_attn_func is not None; print('FA3 OK')"

# Own layer: no wheel for this torch/py, so it builds from source and changes to the bench loop
# below must not recompile it. FLASH_ATTENTION_FORCE_BUILD skips setup.py's search for a release
# wheel, which no release publishes for this torch. --no-deps: torch and einops are in place.
FROM post-fa3 AS fa2
RUN set -e; \
    mkdir -p /tmp/flash-attention \
        && cd /tmp/flash-attention \
        && git init -q \
        && git remote add origin https://github.com/Dao-AILab/flash-attention.git \
        && git fetch --depth 1 origin "${FLASH_ATTENTION_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && git submodule update --init --depth 1 csrc/cutlass; \
    FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install --no-build-isolation --no-deps .; \
    cd / && rm -rf /tmp/flash-attention; \
    python -c "import flash_attn, flash_attn_2_cuda; print('FA2', flash_attn.__version__)"

# Every other version in the image follows vllm's requirements, so its install is fatal; a failed
# one would leave a green image without the stack the lock was resolved for. Its one unsatisfied
# pin, tilelang==0.1.12, is replaced through UV_OVERRIDE. Its wheel carries the image's only
# DeepGEMM, as vllm.third_party.deep_gemm.
FROM fa2 AS fullstack
RUN uv pip install --no-build-isolation vllm
RUN uv pip install --no-build-isolation flash-linear-attention \
    || echo "WARNING: bench baseline 'flash-linear-attention' install failed (non-fatal)"
# Prebuilt here: a first-use JIT build would otherwise land inside a timed benchmark.
RUN DS_BUILD_OPS=0 DS_BUILD_QUANTIZER=1 DS_ACCELERATOR=cuda \
        uv pip install --no-build-isolation deepspeed \
    && python -c "import torch; from deepspeed.ops.quantizer import quantizer_op; assert callable(quantizer_op.quantize)"
# --no-deps: flag_gems 5.0.2 pins numpy, sqlalchemy and packaging to exact versions the lock
# resolves otherwise; verify_runtime_stack.py names those three as the only gaps it may leave.
# sqlalchemy first, since flag_gems imports it at module load.
RUN uv pip install sqlalchemy \
    && uv pip install --no-deps flag_gems \
    || echo "WARNING: bench baseline 'flag_gems' install failed (non-fatal)"
# The Hadamard transform baselines. Both non-fatal, like the baselines above: no test imports
# them, and their absence drops benchmark rows rather than failing the image.
# A commit, not PyPI's 1.1.0 sdist: only v1.1.0.post2 gates `c10::DeviceGuard` by torch version and
# takes the lightweight CUDAGuard header, and both are needed to compile against this torch.
# FORCE_BUILD compiles here instead of fetching a release wheel: the releases carry no
# cu13/torch2.14 build. --no-deps: torch, packaging and ninja are already installed and pinned.
ARG FAST_HADAMARD_TRANSFORM_GIT_SHA=e7706faf8d1c3b9f241e36860640ad1dac644ede
RUN { mkdir -p /tmp/fast-hadamard-transform \
        && cd /tmp/fast-hadamard-transform \
        && git init -q \
        && git remote add origin https://github.com/Dao-AILab/fast-hadamard-transform.git \
        && git fetch --depth 1 origin "${FAST_HADAMARD_TRANSFORM_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE \
            uv pip install --no-build-isolation --no-deps .; } \
    || echo "WARNING: bench baseline 'fast-hadamard-transform' install failed (non-fatal)"; \
    rm -rf /tmp/fast-hadamard-transform
# HadaCore. It is one directory of a monorepo that publishes no release, so the SHA is the last
# commit touching that directory. Its setup.py hard-codes -gencode for sm_80, sm_89 and sm_90 and
# reads no TORCH_CUDA_ARCH_LIST; sm_90 is the pool's arch and is in that list. The package it
# installs is named faster_hadamard_transform.
ARG HADACORE_GIT_SHA=c833061bd85d2e7f3af796b9a461e27dcd73f812
RUN { mkdir -p /tmp/applied-ai \
        && cd /tmp/applied-ai \
        && git init -q \
        && git remote add origin https://github.com/meta-pytorch/applied-ai.git \
        && git fetch --depth 1 origin "${HADACORE_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && uv pip install --no-build-isolation --no-deps \
            ./kernels/cuda/inference/hadamard_transform; } \
    || echo "WARNING: bench baseline 'HadaCore' install failed (non-fatal)"; \
    rm -rf /tmp/applied-ai

# After the bench baselines, so tilelang links against the final stack and a TILELANG_GIT_SHA bump
# reuses every layer before this one. By default it installs the tilelang release constraints.txt
# pins, which vllm already pulled through the override; naming it here states it at this layer.
# TILELANG_GIT_SHA compiles that main commit instead: the wheel replaces the pinned release, so the
# constraints and the override are dropped for that one install.
# `pip wheel` stays on pip — uv has no equivalent subcommand. cutlass and tvm only:
# composable_kernel is the ROCm backend's, and this image builds CUDA.
FROM fullstack AS tilelang
ARG TILELANG_GIT_SHA
RUN if [ -n "${TILELANG_GIT_SHA}" ]; then \
        mkdir -p /tmp/tilelang && cd /tmp/tilelang \
        && git init -q \
        && git remote add origin https://github.com/tile-ai/tilelang.git \
        && git fetch --depth 1 origin "${TILELANG_GIT_SHA}" \
        && git checkout -q FETCH_HEAD \
        && git submodule update --init --depth 1 --recursive --jobs 1 \
            3rdparty/cutlass 3rdparty/tvm \
        && mkdir -p /opt/tilelang-wheels \
        && CMAKE_BUILD_PARALLEL_LEVEL="${MAX_JOBS}" \
            env -u PIP_CONSTRAINT python -m pip wheel . --no-deps --no-build-isolation -w /opt/tilelang-wheels \
        && env -u UV_CONSTRAINT -u UV_OVERRIDE \
            uv pip install --no-deps --reinstall /opt/tilelang-wheels/tilelang-*.whl \
        && rm -rf /tmp/tilelang /opt/tilelang-wheels; \
    else \
        uv pip install tilelang; \
    fi
# DeepSeek's TileLang kernels, a benchmark baseline, linked against the tilelang just installed.
# --no-deps: its two requirements, torch and tilelang>=0.1.15, are in place, and a TILELANG_GIT_SHA
# wheel must not be swapped back for the release. Fatal: the import is the check.
RUN uv pip install --no-deps tile-kernels \
    && python -c "import tile_kernels; print('tile-kernels', tile_kernels.__version__)"
# --no-deps: it must not re-resolve a CUDA wheel under torch.
RUN uv pip install --no-deps cupti-python
# Deps resolved, unlike cupti-python: all four are already pinned in the lock. Fatal, unlike the
# bench baselines: the GEMM benchmark imports it with no fallback, so its absence fails rows.
RUN uv pip install nvmath-python
COPY scripts/ci/verify_runtime_stack.py /tmp/verify_runtime_stack.py
RUN python /tmp/verify_runtime_stack.py
# The image names itself, so a run can report which image produced it. Asking the registry which
# tag is newest answers a different question. -e overrides. Last in the stage, so both targets
# carry it and a tag change invalidates nothing above.
ARG TILEOPS_RUNNER_IMAGE=""
ENV TILEOPS_RUNNER_IMAGE=${TILEOPS_RUNNER_IMAGE}

# The dev image plus the published tileops and tilefoundry releases. tilefoundry caps transformers
# below the lock's pin; the override keeps the lock's version, on top of overrides.txt, which
# --override replaces UV_OVERRIDE with.
FROM tilelang AS release
# Links the ghcr package to this repository.
LABEL org.opencontainers.image.source=https://github.com/tile-ai/TileOPs
ARG TILEOPS_VERSION
ARG TILEFOUNDRY_VERSION
RUN test -n "${TILEOPS_VERSION}" && test -n "${TILEFOUNDRY_VERSION}" \
    && { cat /tmp/overrides.txt; grep '^transformers==' /tmp/constraints-runner-lock.txt; } \
        > /tmp/release-overrides.txt \
    && uv pip install --override /tmp/release-overrides.txt \
        "tileops==${TILEOPS_VERSION}" "tilefoundry==${TILEFOUNDRY_VERSION}" \
    && tilefoundry --version

FROM tilelang AS final
# The host bind-mounts /ci-cache; the dirs are pre-created for unmounted runs.
ENV AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool \
    TILELANG_CACHE_DIR=/ci-cache/tilelang \
    TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
    TRITON_CACHE_DIR=/ci-cache/triton \
    PIP_CACHE_DIR=/ci-cache/pip \
    PIP_NO_CACHE_DIR=0
ARG RUNNER_VERSION=2.338.2
# UID/GID 1000: the account must own the shared /ci-cache bind-mount across hosts. The chown below
# matters only unmounted — in CI the bind-mount shadows those dirs.
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
# Absolute and outside the home dir: exec form expands nothing, a relative path breaks any -w run.
COPY .github/runner/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh
USER ci-runner
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
