#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNNER_TEMP:-}" ]]; then
  echo "::error::RUNNER_TEMP is not set; cannot create a run-local nightly venv"
  exit 1
fi

if [[ -z "${GITHUB_RUN_ID:-}" || -z "${GITHUB_RUN_ATTEMPT:-}" || -z "${GITHUB_JOB:-}" ]]; then
  echo "::error::GitHub run metadata is incomplete; cannot create a scoped nightly venv"
  exit 1
fi

RUNTIME_ROOT="${RUNNER_TEMP}/nightly-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${GITHUB_JOB}"
VENV_PATH="${RUNTIME_ROOT}/venv"

rm -rf "${RUNTIME_ROOT}"
mkdir -p "${RUNTIME_ROOT}"

echo "Creating run-local nightly venv at ${VENV_PATH}"
python -m venv --system-site-packages "${VENV_PATH}"

# shellcheck source=/dev/null
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel --no-user

{
  echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
  echo "VENV_PATH=${VENV_PATH}"
} >> "${GITHUB_ENV}"

echo "Nightly venv ready:"
echo "RUNTIME_ROOT=${RUNTIME_ROOT}"
echo "VENV_PATH=${VENV_PATH}"
python --version
python -m pip --version
