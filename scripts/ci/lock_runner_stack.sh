#!/usr/bin/env bash
# Regenerate constraints-runner-lock.txt: the exact closure of every package pip installs into
# the runner image.
#
# Run from the repository root. Resolution only — nothing is installed. Run it again after
# changing a version in constraints.txt or an install list in the Dockerfile, and read the diff
# before committing: a one-line version bump that moves fifty transitive pins is the signal
# this file exists to surface.
#
# Resolving on the build host emulates the image's interpreter rather than being it. If a
# package turns out to publish a different wheel for the real base image, the build fails on the
# pin instead of installing something else — rerun this inside the base image to settle it:
#
#   docker run --rm -v "$PWD:/src" -w /src nvidia/cuda:13.2.1-devel-ubuntu22.04 \
#       bash -c 'apt-get update -qq && apt-get install -y -qq python3-pip \
#                && scripts/ci/lock_runner_stack.sh'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${REPO_ROOT}/constraints-runner-lock.txt"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

TORCH_INDEX="https://download.pytorch.org/whl/cu132"
TORCH_SPECS=("torch==2.13.0+cu132" "torchvision==0.28.0+cu132")

# Every pip-resolvable package the image installs. The source-built ones (tilelang, FA2, FA3,
# mamba-ssm, DeepGEMM) are absent: they ship no wheel to resolve, install with --no-deps, and
# are held in place by the pins this file produces for everything they link against.
REQUIREMENTS=(
    setuptools wheel ninja scikit-build-core patchelf cmake
    triton apache-tvm-ffi cloudpickle ml_dtypes numpy psutil tqdm
    typing_extensions Cython z3-solver torch_c_dlpack_ext einops "PyYAML>=6.0"
    pytest pytest-xdist ruff pytest-timeout py-spy
    "flash-linear-attention==0.5.2" "vllm==0.27.1" "cupti-python==13.2.0"
    cuda-tile nvidia-cudnn-frontend "nvidia-cutlass-dsl[cu13]==4.6.0" nvidia-ml-py
)

# The wheel tags a manylinux package may carry are not one value: pip matches an explicit
# --platform exactly, so every glibc floor in use has to be listed or a valid wheel reads as
# missing. Harmless when the resolve runs inside the image, where these are unnecessary.
PLATFORMS=()
for glibc in 5 12 17 18 20 23 24 25 26 27 28 31 34 35 36 38 39; do
    PLATFORMS+=(--platform "manylinux_2_${glibc}_x86_64")
done
PLATFORMS+=(--platform manylinux2014_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64)

python3 -m pip install --dry-run --report "${WORK}/report.json" \
    --ignore-installed --no-input --quiet \
    --python-version 3.12 --implementation cp --abi cp312 "${PLATFORMS[@]}" \
    --only-binary=:all: --target "${WORK}/site" \
    --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple \
    -c "${REPO_ROOT}/constraints.txt" \
    "${TORCH_SPECS[@]}" "${REQUIREMENTS[@]}"

OUT="${OUT}" python3 - "${WORK}/report.json" <<'PY'
import json
import os
import sys

report = json.load(open(sys.argv[1]))
pins = []
for item in report["install"]:
    name = item["metadata"]["name"]
    version = item["metadata"]["version"]
    if name.lower() == "tilelang":
        continue
    url = item.get("download_info", {}).get("url", "")
    if "+cu132" in url or "%2Bcu132" in url:
        # The local version segment is what makes it the CUDA build rather than the PyPI one.
        version = version if "+" in version else version + "+cu132"
    pins.append((name, version))
pins.sort(key=lambda pair: pair[0].lower())

header = """\
# Generated closure of the runner image's pip stack — DO NOT EDIT BY HAND.
#
# Every package pip resolves for the image, pinned exactly, so no install step can move a
# version another step already settled on. constraints.txt states which versions we choose and
# why; this file is the transitive closure of that choice, and the Dockerfile passes both to
# every pip install.
#
# tilelang is deliberately absent: it is compiled from source in its own stage, and a pin here
# would reject the wheel that stage builds. The build-time guard checks its ABI coupling to
# apache-tvm-ffi instead.
#
# Regenerate with scripts/ci/lock_runner_stack.sh after changing anything in constraints.txt
# or the Dockerfile's install lists, and re-read the diff before committing it.
"""
with open(os.environ["OUT"], "w") as handle:
    handle.write(header)
    handle.writelines(f"{name}=={version}\n" for name, version in pins)
print(f"{len(pins)} pins -> {os.environ['OUT']}")
PY
