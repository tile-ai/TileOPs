#!/usr/bin/env bash
# Prepare CI cache directories and fail early when any selected cache root is
# not writable. Workflows may opt into run-local Triton/TorchInductor caches to
# avoid stale shared-cache subdirectories with mismatched ownership.
set -euo pipefail

sanitize_scope() {
  local raw="$1"
  # Keep paths readable while avoiding accidental separators or shell syntax.
  printf '%s' "${raw}" | tr -c 'A-Za-z0-9._-' '-'
}

append_env() {
  local key="$1"
  local value="$2"
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    echo "${key}=${value}" >> "${GITHUB_ENV}"
  fi
  export "${key}=${value}"
}

set_current_env() {
  local key="$1"
  local value="$2"
  export "${key}=${value}"
}

require_env() {
  local key="$1"
  local value="${!key:-}"
  if [[ -z "${value}" ]]; then
    echo "::error::${key} is not set; CI cache layout is incomplete"
    exit 1
  fi
}

prepare_dir() {
  local label="$1"
  local dir="$2"
  if [[ -z "${dir}" ]]; then
    return 0
  fi

  if ! mkdir -p "${dir}" 2>/dev/null; then
    echo "::error::${label} could not be created: ${dir}"
    ls -ld "$(dirname "${dir}")" 2>/dev/null || true
    exit 1
  fi
  chmod u+rwX "${dir}" 2>/dev/null || true

  if [[ ! -d "${dir}" || ! -w "${dir}" ]]; then
    echo "::error::${label} is not writable: ${dir}"
    ls -ld "${dir}" 2>/dev/null || true
    exit 1
  fi

  local probe
  if ! probe="$(mktemp "${dir}/.ci-write-test.XXXXXX" 2>/dev/null)"; then
    echo "::error::${label} is not writable (probe file creation failed): ${dir}"
    ls -ld "${dir}" 2>/dev/null || true
    exit 1
  fi
  rm -f "${probe}"
}

require_env TILELANG_CACHE_DIR
require_env TILELANG_TMP_DIR
require_env TRITON_CACHE_DIR

scope="${CI_CACHE_RUN_SCOPE:-${GITHUB_WORKFLOW:-ci}-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}}"
scope="$(sanitize_scope "${scope}")"

if [[ "${CI_USE_RUN_LOCAL_TRITON_CACHE:-0}" == "1" ]]; then
  triton_root="${TRITON_CACHE_ROOT:-${TRITON_CACHE_DIR}}"
  set_current_env TRITON_CACHE_DIR "${triton_root%/}/${scope}"
fi

if [[ "${CI_USE_RUN_LOCAL_TORCHINDUCTOR_CACHE:-0}" == "1" ]]; then
  inductor_root="${TORCHINDUCTOR_CACHE_ROOT:-${TORCHINDUCTOR_CACHE_DIR:-/ci-cache/torchinductor}}"
  set_current_env TORCHINDUCTOR_CACHE_DIR "${inductor_root%/}/${scope}"
elif [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
  set_current_env TORCHINDUCTOR_CACHE_DIR "${TORCHINDUCTOR_CACHE_DIR}"
fi

prepare_dir TILELANG_CACHE_DIR "${TILELANG_CACHE_DIR}"
prepare_dir TILELANG_TMP_DIR "${TILELANG_TMP_DIR}"
prepare_dir TRITON_CACHE_DIR "${TRITON_CACHE_DIR}"
prepare_dir PIP_CACHE_DIR "${PIP_CACHE_DIR:-}"
prepare_dir WHEEL_DIR "${WHEEL_DIR:-}"
prepare_dir TORCHINDUCTOR_CACHE_DIR "${TORCHINDUCTOR_CACHE_DIR:-}"
prepare_dir XDG_CACHE_HOME "${XDG_CACHE_HOME:-}"

for key in TILELANG_CACHE_DIR TILELANG_TMP_DIR TRITON_CACHE_DIR PIP_CACHE_DIR WHEEL_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME; do
  value="${!key:-}"
  if [[ -n "${value}" ]]; then
    append_env "${key}" "${value}"
  fi
done

echo "CI cache directories:"
for key in TILELANG_CACHE_DIR TILELANG_TMP_DIR TRITON_CACHE_DIR PIP_CACHE_DIR WHEEL_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME; do
  value="${!key:-}"
  if [[ -n "${value}" ]]; then
    echo "${key}=${value}"
  fi
done
