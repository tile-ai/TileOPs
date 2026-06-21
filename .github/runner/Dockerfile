# syntax=docker/dockerfile:1.7
# Multi-stage TileOPs CI runner image. Built in the image-build env;
# published to ghcr.io. From a PUBLIC base — fully reproducible. NOT TileOPs source.
ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu24.04

# Public base: Ubuntu 24.04 (native python3.12) + CUDA 12.9 devel toolkit (nvcc is
# needed at runtime for tilelang JIT, so we keep devel, not runtime). torch cu129
# matches the base CUDA 12.9 → no LD_LIBRARY_PATH hack. Driver 575.57.08 caps CUDA at
# 12.9 (13.x needs 580+). No DOCA/IB (single-node NCCL uses NVLink/P2P).

# ── Stage: runtime — python3.12 + torch cu129 + deps under constraints (einops/PyYAML float) + tilelang built once from SHA ──
FROM ${BASE_IMAGE} AS runtime
# 24.04 ships python3.12 natively. `python-is-python3` gives a `python` symlink
# (later stages call `python`); PIP_BREAK_SYSTEM_PACKAGES bypasses PEP 668 so
# system-wide pip installs work in this single-purpose image.
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip python-is-python3 git curl ca-certificates \
        build-essential cmake ninja-build unzip xz-utils sudo \
    && rm -rf /var/lib/apt/lists/*
COPY constraints.txt /tmp/constraints.txt
# torch from the cu129 index → 2.10.0+cu129, aligned with the base CUDA 12.9. The
# `+cu129` local-version pin forces the cu129 wheel from the pytorch index; the
# PyPI extra-index supplies torch's pure-python transitive deps (sympy, networkx,
# jinja2, filelock, fsspec, typing_extensions), which the pytorch index does not host.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu129 \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/constraints.txt "torch==2.10.0+cu129"
RUN python -m pip install -c /tmp/constraints.txt \
        setuptools wheel ninja \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"

# tilelang: main mode builds the wheel once from the pinned commit; the wheel is
# kept at /opt/tilelang-wheels and installed --no-deps. For release mode, build the
# image against a release tag instead (and skip this source build).
ARG TILELANG_GIT_SHA=65dbc9837beedf6882a40a08e18ea571d92fd6a5
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
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

# ── Stage: post-fa3 — test tooling + FA3 (pip first, vendor fallback) ──
FROM runtime AS post-fa3
RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt \
        "pytest==9.0.2" "pytest-xdist>=3.0" "ruff==0.14.13"
# FA3: prefer the pip wheel; if it cannot install on this stack, Task 4 replaces
# this line with a vendored COPY + `python setup.py install` (hopper).
RUN python -m pip install --no-cache-dir --no-build-isolation "flash-attn-interface>=2.8.3" \
    || echo "WARNING: flash-attn-interface pip install failed — see Task 4 (vendor fallback)"

# ── Stage: fullstack — bench baselines (pip subset + vendored remainder) ──
FROM post-fa3 AS fullstack
RUN python -m pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt \
        "flash-attn==2.8.3" \
        "flash-linear-attention==0.4.2" \
        "flashinfer>=0.6.6" \
        "vllm==0.18.0" \
        "sgl-kernel==0.3.21" \
    || echo "WARNING: a bench pip baseline failed — see Task 4"
# Vendored remainder (cannot pip-install): mamba-ssm, native-sparse-attention,
# and any pip baseline that failed above. Task 4 finalizes this set and the COPY
# lines. Build context must contain .github/runner/vendor/<dep> (Prerequisites).
# COPY .github/runner/vendor/mamba-ssm /tmp/vendor/mamba-ssm
# RUN MAMBA_FORCE_BUILD=TRUE python -m pip install --no-cache-dir --no-build-isolation /tmp/vendor/mamba-ssm
# COPY .github/runner/vendor/native-sparse-attention /tmp/vendor/native-sparse-attention
# RUN python -m pip install --no-cache-dir --no-build-isolation /tmp/vendor/native-sparse-attention

# ── Stage: final — GitHub Actions runner; NO TileOPs source ──
FROM fullstack AS final
# Cache dirs as image ENV defaults: workflows/scripts read these env vars rather than
# hardcoding the path. The host persistent cache dir is bind-mounted to /ci-cache at
# container start; the dirs are also pre-created below (owned by ci-runner) so the
# defaults stay writable even when the image runs without that mount.
ENV AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool \
    TILELANG_CACHE_DIR=/ci-cache/tilelang \
    TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
    TRITON_CACHE_DIR=/ci-cache/triton \
    PIP_CACHE_DIR=/ci-cache/pip \
    PIP_NO_CACHE_DIR=0
# Build stages set PIP_NO_CACHE_DIR=1 to keep image layers slim; re-enable caching
# here so CI-runtime pip installs reuse the bind-mounted PIP_CACHE_DIR.
ARG RUNNER_VERSION=2.334.0
# Create the user (and its skeleton home) before WORKDIR so the home dir is owned by
# ci-runner, not root. WORKDIR then sits inside that home and precedes the relative-path
# runner-extraction RUN (hadolint requires WORKDIR before relative paths).
RUN useradd -m -s /bin/bash ci-runner
WORKDIR /home/ci-runner/runner
RUN mkdir -p /home/ci-runner/runner/_work/_tool \
    && curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh \
    && mkdir -p /ci-cache/pip /ci-cache/tilelang/tmp /ci-cache/triton \
    && chown -R ci-runner:ci-runner /home/ci-runner /ci-cache
# Standalone runner mode: entrypoint.sh configures an ephemeral runner (config.sh)
# then runs one job (run.sh). chmod needs root, so set it before dropping to ci-runner.
COPY --chown=ci-runner:ci-runner .github/runner/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh
USER ci-runner
ENTRYPOINT ["./entrypoint.sh"]
