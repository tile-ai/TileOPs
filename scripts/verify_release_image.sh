#!/usr/bin/env bash
# Pull the release image and verify the tileops it installs, on this machine's GPU.
#
#   scripts/verify_release_image.sh [--image <ref>] [--gpus <docker --gpus value>] \
#       [--bench <path under benchmarks/>]... [--no-pull]
#
# --no-pull verifies a locally built image before it is pushed.
#
# Tests and benchmarks are not in the wheel, so they come from this repository at the tag
# of the image's tileops version. The src layout keeps them on the installed package.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="https://github.com/tile-ai/TileOPs"

IMAGE="ghcr.io/tile-ai/tileops-release:tileops0.1.0-tilefoundry0.0.3-cuda13.2-torch2.13-tilelang0.1.12"
GPUS="all"
BENCH=()
PULL=1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image) IMAGE="${2:?--image needs a value}"; shift 2 ;;
    --gpus) GPUS="${2:?--gpus needs a value}"; shift 2 ;;
    --bench) BENCH+=("${2:?--bench needs a value}"); shift 2 ;;
    --no-pull) PULL=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "${PULL}" = 1 ]; then
  echo "=== Pull ${IMAGE} ==="
  docker pull "${IMAGE}"
fi

VERSION="$(docker run --rm --entrypoint python "${IMAGE}" -c \
  'import importlib.metadata as m; print(m.version("tileops"))' | tail -n 1)"
TAG="v${VERSION}"
echo "=== Image installs tileops ${VERSION}; tests come from ${TAG} ==="

if ! git -C "${REPO_ROOT}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  git -C "${REPO_ROOT}" fetch --quiet --depth 1 "${UPSTREAM}" "refs/tags/${TAG}:refs/tags/${TAG}"
fi
SRC="$(mktemp -d "${TMPDIR:-/tmp}/tileops-release-verify.XXXXXX")"
trap 'rm -rf "${SRC}"' EXIT
git -C "${REPO_ROOT}" archive "refs/tags/${TAG}" | tar -x -C "${SRC}"

# The tree is copied inside the container: tests and benchmarks write into it, never to the host.
in_image() {
  docker run --rm --gpus "${GPUS}" -v "${SRC}:/src:ro" "${IMAGE}" \
    bash -c 'cp -r /src /workspace && cd /workspace && "$@"' _ "$@"
}

echo "=== Runtime stack and baselines ==="
in_image python scripts/ci/verify_runner_image.py
in_image python -c 'import fla, vllm, fast_hadamard_transform, faster_hadamard_transform'
in_image tilefoundry --version

echo "=== Packaging tests ==="
in_image python -c 'import tileops; print(tileops.__version__, tileops.__file__)'
in_image python -m pytest -q tests/ops tests/test_installed_surface.py -m packaging

for bench in "${BENCH[@]}"; do
  echo "=== Benchmark ${bench} ==="
  in_image python -m pytest -q "${bench}"
done

echo "Release image ${IMAGE} verified."
