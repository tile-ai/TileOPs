# syntax=docker/dockerfile:1.7
# Multi-stage CI runner image for the self-hosted GPU runner.
# 13.2.1, not a later 13.x: cu132 is the newest CUDA variant the PyTorch index publishes, and
# matching the base image to the torch wheels keeps one CUDA minor across nvcc, the runtime
# libraries torch bundles, and every source-built extension.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu22.04

# ── runtime ──
# devel (not runtime) base: nvcc is needed both at build time (FA3 / tilelang source builds)
# and at runtime (tilelang JIT).
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
# torch / torchvision: cu132 wheels from the pytorch index (the PyPI default is a different
# CUDA variant, so pin +cu132). Installing torchvision as cu132 here means vllm later finds its
# exact-pinned version already satisfied and never swaps it off cu132. torchaudio is absent on
# purpose: the cu132 index carries no build of it, and TileOPs imports nothing from it. vllm
# pins torchaudio==2.11.0, whose wheel declares no dependencies, so pip picks that up from PyPI
# later without touching torch.
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu132 \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/constraints.txt \
        "torch==2.13.0+cu132" "torchvision==0.28.0+cu132"
# tilelang's build backend (scikit-build-core + patchelf, its [build-system].requires) and its
# runtime deps, installed here so the tilelang stage can build with --no-build-isolation. cmake
# from pip provides >=3.26.1 (tilelang's [tool.scikit-build] floor); jammy apt cmake is 3.22.
RUN python -m pip install -c /tmp/constraints.txt \
        setuptools wheel ninja scikit-build-core patchelf cmake \
        triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm \
        typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"
ARG MAX_JOBS=64
ARG NVCC_THREADS=4
# TORCH_CUDA_ARCH_LIST=9.0: the runner pool is Hopper (H200) only. Drives the FA3 and tilelang
# source builds in later stages, and runtime tilelang JIT.
# CUDA_HOME / PATH are pinned to the base image's toolkit on purpose: bench baselines pull
# `nvidia-cuda-nvcc` wheels of a LATER CUDA minor into site-packages, and a source build or a
# JIT that picked those up would emit code the image's own runtime libraries do not match.
ENV MAX_JOBS=${MAX_JOBS} NVCC_THREADS=${NVCC_THREADS} TORCH_CUDA_ARCH_LIST=9.0 \
    CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:${PATH}

# ── post-fa3 ──
# Test tooling + FlashAttention-3 (a bench baseline). tilelang is NOT built here — it is built
# last (the `tilelang` stage) so a SHA bump rebuilds only that layer and so it links against
# the final post-bench stack.
FROM runtime AS post-fa3
RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt \
        pytest pytest-xdist ruff pytest-timeout py-spy
# FlashAttention-3 (Hopper) has no PyPI wheel — build it from the repo's hopper/ dir.
# Fetch ONLY the csrc/cutlass submodule the build needs: a full --recursive also pulls the AMD
# composable_kernel submodule, which the sm_90 build never uses and which is large and
# slow/timeout-prone to clone. Best-effort: a bench baseline must not break the image. The
# source tree is removed whether or not the build succeeds, so it never lands in the image.
RUN git clone --depth 1 --branch v2.8.3 \
        https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention \
        && cd /tmp/flash-attention \
        && git submodule update --init --depth 1 csrc/cutlass \
        && cd hopper && python setup.py install; \
    status=$?; rm -rf /tmp/flash-attention; \
    [ "$status" -eq 0 ] || echo "WARNING: FlashAttention-3 (hopper) build failed (non-fatal)"

# ── fa2 ──
# FlashAttention-2 in its OWN stage/layer: no prebuilt wheel for this torch/cu132/py, so it
# builds from source (~minutes). Isolating it means changes to the bench-install loop
# (fullstack) never invalidate this expensive compile. flash-attn does not re-resolve torch.
# Best-effort: a bench baseline must not fail the image.
FROM post-fa3 AS fa2
RUN python -m pip install --no-cache-dir --no-build-isolation \
        --index-url https://download.pytorch.org/whl/cu132 \
        --extra-index-url https://pypi.org/simple \
        -c /tmp/constraints.txt "flash-attn==2.8.3.post1" \
    || echo "WARNING: flash-attn (FA2) install failed (non-fatal)"

# ── fullstack ──
# Remaining bench baselines (comparison targets). Best-effort. Freeze what `runtime` installed
# into a constraint first so a bench dep's unversioned requirement cannot swap it: the torch
# packages themselves, and — new on the CUDA 13 line — every CUDA runtime library torch pulls
# in, since replacing one of those breaks the c10 ABI just as thoroughly as replacing torch.
# Two naming schemes have to be caught: the `nvidia-*` wheels, and the unsuffixed
# `cuda-toolkit` / `cuda-bindings` / `cuda-pathfinder` that torch 2.13 depends on directly.
# constraints.txt itself stays bare so it also fits the CPU preflight. vllm pins an exact
# flashinfer (upgraded below) and has no apache-tvm-ffi pin, so it does not conflict with
# tilelang. sgl-kernel is not installed.
FROM fa2 AS fullstack
RUN python -m pip freeze \
        | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' \
        > /tmp/torch-pin.txt \
    && for pkg in \
        "flash-linear-attention==0.5.2" \
        "vllm==0.27.1"; do \
        python -m pip install --no-cache-dir --no-build-isolation \
            --index-url https://download.pytorch.org/whl/cu132 \
            --extra-index-url https://pypi.org/simple \
            -c /tmp/constraints.txt -c /tmp/torch-pin.txt "$pkg" \
            || echo "WARNING: bench baseline '$pkg' install failed (non-fatal)"; \
    done
# The Mamba benchmarks compare against the official mamba_ssm Triton kernels. Keep this
# best-effort and dependency-free so it cannot re-resolve the CUDA-specific torch/triton stack
# or pull the newer mamba_ssm tilelang/apache-tvm-ffi pins into the runner image. mamba_ssm's
# package import loads selective_scan_cuda, so the CUDA extension must be built even when the
# benchmark uses the Triton modules.
RUN MAMBA_FORCE_BUILD=TRUE \
    python -m pip install --no-cache-dir --no-build-isolation --no-deps \
        -c /tmp/constraints.txt -c /tmp/torch-pin.txt \
        "mamba-ssm==2.3.2.post1" \
    || echo "WARNING: mamba-ssm 2.3.2.post1 install failed (non-fatal)"
# The grouped-GEMM benchmark can compare against DeepGEMM. Install from the official tag
# instead of PyPI's stale 1.0.0 sdist, and fetch only the submodules needed for its headers.
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
        && DG_FORCE_BUILD=1 python -m pip install --no-cache-dir --no-build-isolation --no-deps \
            -c /tmp/constraints.txt -c /tmp/torch-pin.txt .; } \
    || echo "WARNING: DeepGEMM v2.1.1.post3 install failed (non-fatal)"; \
    rm -rf /tmp/DeepGEMM
# The attention benchmarks (benchmarks/ops/attention/) need a flashinfer newer than the
# 0.6.16.post3 vllm pins. Upgrade flashinfer-python/-cubin with --no-deps so torch and vllm's
# other deps are untouched; add plain cuda-tile (flashinfer needs it). nvidia-cutlass-dsl takes
# vllm's exact pin and its cu13 extra, so this step cannot pull a second CUTLASS build. vllm's
# flashinfer pin is left unsatisfied, which is safe: vllm's only TileOPs-used path (fused_moe)
# does not import flashinfer.
RUN { python -m pip install --no-cache-dir --no-deps -c /tmp/constraints.txt -c /tmp/torch-pin.txt \
            "flashinfer-python==0.6.17" "flashinfer-cubin==0.6.17" \
        && python -m pip install --no-cache-dir -c /tmp/constraints.txt -c /tmp/torch-pin.txt \
            cuda-tile nvidia-cudnn-frontend "nvidia-cutlass-dsl[cu13]==4.6.0" nvidia-ml-py; } \
    || echo "WARNING: flashinfer 0.6.17 upgrade failed (non-fatal)"

# ── tilelang ──
# Built LAST (after all bench): a TILELANG_GIT_SHA bump then rebuilds only this layer (bench
# stays cached), and tilelang compiles/links against the exact final stack so its ABI always
# matches what ships. ARG is declared here — not earlier — so changing it never invalidates
# the bench layers above. Building last also settles the one conflict vllm introduces: vllm
# depends on an exact tilelang release, so `fullstack` leaves a PyPI tilelang installed, and
# the wheel built here replaces it. Leaving vllm's pin unsatisfied is the intended outcome —
# the image exists to test the tilelang commit under evaluation, not vllm's.
FROM fullstack AS tilelang
ARG TILELANG_GIT_SHA
ARG TILELANG_VERSION
# main mode:    --build-arg TILELANG_GIT_SHA=<commit>  → clone + compile that commit
# release mode: --build-arg TILELANG_VERSION=<version> → pip install the PyPI release
# Either way --no-deps, so pip never re-resolves the cu132 stack.
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
        && python -m pip install --no-deps /opt/tilelang-wheels/tilelang-*.whl \
        && rm -rf /tmp/tilelang; \
    elif [ -n "${TILELANG_VERSION}" ]; then \
        python -m pip install --no-deps "tilelang==${TILELANG_VERSION}"; \
    else \
        echo "ERROR: set --build-arg TILELANG_GIT_SHA=<commit> (main) or TILELANG_VERSION=<ver> (release)"; \
        exit 1; \
    fi
# CUPTI binding for benchmark timing, matched to the base image's CUDA minor — it requires
# nvidia-cuda-cupti~=13.2, which is what torch installs here. --no-deps regardless, so it can
# never re-resolve a CUDA runtime wheel out from under torch; a resolved install that moved
# cuda-bindings off torch's pin would only warn, and the guard below is what catches it.
RUN python -m pip install --no-cache-dir --no-deps "cupti-python==13.2.0"
# Guard the final stack (GPU-free, runs at build): tilelang must import, the installed
# apache-tvm-ffi must sit in the tilelang wheel's declared ABI range, torch must still be
# the cu132 build, and cupti-python must not have dragged cuda-bindings off torch's pin.
COPY scripts/ci/verify_runtime_stack.py /tmp/verify_runtime_stack.py
RUN python /tmp/verify_runtime_stack.py

# ── final ──
FROM tilelang AS final
# Cache dir defaults (host bind-mounts /ci-cache at runtime; pre-created below so they stay
# writable when run unmounted). PIP_NO_CACHE_DIR=0 re-enables caching (build stages disable it).
ENV AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool \
    TILELANG_CACHE_DIR=/ci-cache/tilelang \
    TILELANG_TMP_DIR=/ci-cache/tilelang/tmp \
    TRITON_CACHE_DIR=/ci-cache/triton \
    PIP_CACHE_DIR=/ci-cache/pip \
    PIP_NO_CACHE_DIR=0
ARG RUNNER_VERSION=2.336.0
# The /ci-cache chown below serves only the unmounted image-verification path
# (`docker run <image> python ...`, see README): ci-runner writes the root-created cache dirs
# directly. In the mounted CI path the host bind-mount shadows these dirs, so the chown is a
# no-op and host ownership wins.
# useradd before WORKDIR: home stays ci-runner-owned, and WORKDIR precedes the relative-path RUN (hadolint).
# Pin UID/GID 1000 so the account deterministically owns the shared /ci-cache bind-mount across
# hosts (it matches the majority of existing warm-cache entries; host normalization preserves them).
# FlashInfer's flashinfer-cubin package keeps its packaged cubin tree under site-packages.
# Some baselines materialize TRTLLM GEMM entries there at runtime, so the unprivileged runner
# must own the writable directories instead of inheriting pip's root-owned directories.
RUN groupadd -g 1000 ci-runner && useradd -u 1000 -g 1000 -m -s /bin/bash ci-runner \
    && FLASHINFER_CUBIN_DIR="$(python -c 'import pathlib, flashinfer_cubin; print(pathlib.Path(flashinfer_cubin.get_cubin_dir()))' 2>/dev/null || true)" \
    && if [ -n "${FLASHINFER_CUBIN_DIR}" ]; then \
        mkdir -p "${FLASHINFER_CUBIN_DIR}/flashinfer/trtllm/gemm"; \
        find "${FLASHINFER_CUBIN_DIR}" -type d -exec chown ci-runner:ci-runner {} +; \
    fi
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
