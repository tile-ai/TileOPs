FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl wget git rsync jq \
        build-essential cmake libedit-dev libxml2-dev libtinfo-dev zlib1g-dev \
        software-properties-common ca-certificates gnupg sudo unzip tar gzip \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 ──────────────────────────────────────────────────────────────
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── CI runner user & directories ─────────────────────────────────────────────
RUN useradd -m -s /bin/bash ci-runner \
    && mkdir -p /home/ci-runner/.tilelang/cache \
               /home/ci-runner/.triton/cache \
               /home/ci-runner/.cache/pip \
               /home/ci-runner/.wheel-cache \
               /home/ci-runner/runner/_work/_tool \
    && chown -R ci-runner:ci-runner /home/ci-runner

# ── GitHub Actions Runner ────────────────────────────────────────────────────
ARG RUNNER_VERSION=2.333.0
RUN curl -o /tmp/runner.tar.gz -L \
        https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && cd /home/ci-runner/runner \
    && tar xzf /tmp/runner.tar.gz \
    && rm /tmp/runner.tar.gz \
    && ./bin/installdependencies.sh \
    && chown -R ci-runner:ci-runner /home/ci-runner/runner

ENV MAX_JOBS=64 \
    AGENT_TOOLSDIRECTORY=/home/ci-runner/runner/_work/_tool

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "torch>=2.1.0,<2.11.0" \
        "tilelang==0.1.8" \
        einops numpy "tqdm>=4.62.3" "typing_extensions>=4.10.0" \
        cloudpickle ml_dtypes psutil Cython \
        setuptools wheel ninja

RUN pip install --no-cache-dir \
        "pytest==9.0.2" \
        "pytest-xdist>=3.0" \
        "pyyaml>=6.0"

RUN MAX_JOBS=96 NVCC_THREADS=96 pip install --no-cache-dir --no-build-isolation "flash-attn==2.8.3" \
    || echo "WARNING: flash-attn failed to install -- non-fatal"

RUN MAX_JOBS=96 NVCC_THREADS=96 pip install --no-cache-dir --no-build-isolation "flash-attn-interface>=2.7.2" \
    || echo "WARNING: flash-attn-interface (FA3) failed to install -- non-fatal"

RUN MAX_JOBS=96 NVCC_THREADS=96 pip install --no-cache-dir --no-build-isolation \
        "flash-linear-attention==0.4.2" \
        "flashinfer>=0.6.6" \
        "vllm==0.18.0" \
        "sgl-kernel==0.3.21" \
    || echo "WARNING: Some bench deps failed to install -- non-fatal for runner image"

RUN MAX_JOBS=32 pip install --no-cache-dir \
        "git+https://github.com/fla-org/native-sparse-attention.git@bd67af59b90afa34b25f61d2922e612d10dba3bd" \
    || echo "WARNING: native-sparse-attention failed to install -- non-fatal"

# ── Install TileOPs (editable) ───────────────────────────────────────────────
WORKDIR /home/ci-runner/tileops
COPY --chown=ci-runner:ci-runner . .
RUN pip install --no-cache-dir --no-deps -e .

# ── Entrypoint (for standalone runner mode) ──────────────────────────────────
COPY --chown=ci-runner:ci-runner .github/runner/entrypoint.sh /home/ci-runner/runner/entrypoint.sh
RUN chmod +x /home/ci-runner/runner/entrypoint.sh

USER ci-runner
WORKDIR /home/ci-runner/runner
ENTRYPOINT ["./entrypoint.sh"]
