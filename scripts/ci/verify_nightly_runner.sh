#!/usr/bin/env bash
set -euo pipefail

required_vars=(
  TILELANG_CACHE_DIR
  TILELANG_TMP_DIR
  TRITON_CACHE_DIR
  PIP_CACHE_DIR
  WHEEL_DIR
)

for var in "${required_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "::error::${var} is not set; nightly runner container is not configured correctly"
    exit 1
  fi
done

mkdir -p \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${WHEEL_DIR}"

for dir in \
  "${TILELANG_CACHE_DIR}" \
  "${TILELANG_TMP_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${PIP_CACHE_DIR}" \
  "${WHEEL_DIR}"; do
  if [[ ! -w "${dir}" ]]; then
    echo "::error::${dir} is not writable by the nightly runner"
    exit 1
  fi
done

nvidia-smi -L

echo "Nightly runner environment:"
echo "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR}"
echo "TILELANG_TMP_DIR=${TILELANG_TMP_DIR}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "PIP_CACHE_DIR=${PIP_CACHE_DIR}"
echo "WHEEL_DIR=${WHEEL_DIR}"
echo "MAX_JOBS=${MAX_JOBS:-}"
