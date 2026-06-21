#!/usr/bin/env bash
# Install tileops without letting pip re-resolve tilelang.
#
# Contract: tilelang must already be present — baked into
# the runner image (CI), or installed by the developer (local) — BEFORE this runs.
# tileops is then installed --no-deps under the version lock. pip must never resolve
# tilelang transitively: it would drift the cu129 stack.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints.txt"

if ! python3 -c "import tilelang" >/dev/null 2>&1; then
  {
    echo "::error::tilelang is not importable. Provision it first, then re-run:"
    echo "  release:  pip install --no-deps -c \"${CONSTRAINTS}\" tilelang==0.1.9"
    echo "  main:     pip install --no-deps <prebuilt main-commit tilelang wheel>"
    echo "(the CI runner image bakes tilelang; this script never installs it)"
  } >&2
  exit 1
fi

# tileops itself only. Its runtime deps (torch/einops/pyyaml) and any dev/bench
# extras are provided by the runner image (or installed separately for local dev).
# constraints.txt version-locks the CI/runner stack it covers (torch, triton,
# apache-tvm-ffi, and tilelang's runtime deps); --no-deps keeps pip away from tilelang.
python3 -m pip install -e "${REPO_ROOT}" --no-deps -c "${CONSTRAINTS}"

python3 -c "import tileops, tilelang; print('install_tileops: tileops + tilelang import OK')"
