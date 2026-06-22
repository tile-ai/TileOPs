#!/usr/bin/env bash
# Install tileops with --no-deps so pip never re-resolves tilelang (which would drift the
# cu129 stack). tilelang must already be present — baked into the runner image, or
# installed by the developer locally — before this runs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints.txt"

if ! python3 -c "import tilelang" >/dev/null 2>&1; then
  {
    echo "::error::tilelang is not importable. Provision it first, then re-run:"
    echo "  release:  python3 -m pip install --no-deps -c \"${CONSTRAINTS}\" tilelang==<version>"
    echo "  main:     python3 -m pip install --no-deps <prebuilt main-commit tilelang wheel>"
    echo "(the CI runner image bakes tilelang; this script never installs it)"
  } >&2
  exit 1
fi

# tileops only; its runtime deps come from the runner image.
python3 -m pip install -e "${REPO_ROOT}" --no-deps -c "${CONSTRAINTS}"

python3 -c "import tileops, tilelang; print('install_tileops: tileops + tilelang import OK')"
