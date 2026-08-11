#!/usr/bin/env bash
# Regenerate constraints-runner-lock.txt: the exact closure of every package pip installs into
# the runner image. Resolution only — nothing is installed. Run from the repository root after
# changing a version in constraints.txt or an install list in the Dockerfile, and read the diff.
#
# Needs uv (https://docs.astral.sh/uv/): curl -LsSf https://astral.sh/uv/install.sh | sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${REPO_ROOT}/constraints-runner-lock.txt"
RESOLVED="$(mktemp)"
trap 'rm -f "${RESOLVED}"' EXIT

# Every pip-resolvable package the image installs, matching the Dockerfile's install lists. The
# source-built ones (tilelang, FA2, FA3, mamba-ssm, DeepGEMM) are absent: they ship no wheel to
# resolve and install with --no-deps.
#
# --index-strategy unsafe-best-match matches how pip searches both indexes in the Dockerfile;
# uv's default would take a package from the pytorch index whenever it appears there at all.
# --no-annotate keeps the file one line per pin; drop it to record which package pulled in what.
uv pip compile - --output-file "${RESOLVED}" \
    --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --constraints "${REPO_ROOT}/constraints.txt" \
    --no-annotate --no-header --quiet <<'REQUIREMENTS'
torch==2.13.0+cu132
torchvision==0.28.0+cu132
setuptools
wheel
ninja
scikit-build-core
patchelf
cmake
triton
apache-tvm-ffi
cloudpickle
ml_dtypes
numpy
psutil
tqdm
typing_extensions
Cython
z3-solver
torch_c_dlpack_ext
einops
PyYAML>=6.0
pytest
pytest-xdist
ruff
pytest-timeout
py-spy
flash-linear-attention==0.5.2
vllm==0.27.1
cupti-python==13.2.0
REQUIREMENTS

# tilelang is source-built in its own stage; a pin here would reject that wheel.
{
    cat <<'HEADER'
# Generated closure of the runner image's pip stack — DO NOT EDIT BY HAND.
# Regenerate with scripts/ci/lock_runner_stack.sh; see .github/runner/README.md.
# tilelang is absent on purpose: it is source-built in its own stage, and a pin here would
# reject that wheel.
HEADER
    grep -v '^tilelang==' "${RESOLVED}"
} > "${OUT}"
echo "$(grep -c '^[^#]' "${OUT}") pins -> ${OUT}"
