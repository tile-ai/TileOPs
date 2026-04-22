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
  val="${!var:-}"
  if [[ -z "${val}" ]]; then
    echo "::error::${var} is not set; nightly runner container is not configured correctly"
    exit 1
  fi

  mkdir -p "${val}"
  if [[ ! -w "${val}" ]]; then
    echo "::error::${val} (from ${var}) is not writable by the nightly runner"
    exit 1
  fi
done

nvidia-smi -L

echo "Nightly runner environment:"
for var in "${required_vars[@]}"; do
  echo "${var}=${!var}"
done
echo "MAX_JOBS=${MAX_JOBS:-}"
