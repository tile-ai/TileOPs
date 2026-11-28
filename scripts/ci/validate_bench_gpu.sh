#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "::error::nvidia-smi is not available on this runner"
  exit 1
fi

# The benchmark needs exclusive access to one card. A second visible
# GPU means the container is misprovisioned, and work landing on it
# perturbs these timings.
# Count non-blank rows rather than newlines: output without a trailing
# newline undercounts, which would let a second card pass. Counted in
# the shell so gpu_count is always a number — a pipeline here could
# fail into an empty value, and the comparison below would then error
# out inside an `if`, where set -e does not fire, and pass the guard.
if ! gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader); then
  echo "::error::nvidia-smi failed to report GPU indices"
  exit 1
fi
gpu_count=0
while IFS= read -r gpu_row; do
  if [ -n "${gpu_row//[[:space:]]/}" ]; then
    gpu_count=$((gpu_count + 1))
  fi
done <<< "${gpu_indices}"
if [ "${gpu_count}" -ne 1 ]; then
  echo "::error::expected exactly 1 visible GPU, found ${gpu_count}"
  nvidia-smi --query-gpu=index,name --format=csv,noheader
  exit 1
fi

# The runner is ephemeral, so this log is the only record of which
# card the job got.
echo "GPU under test:"
nvidia-smi --query-gpu=uuid,compute_mode,memory.used --format=csv

compute_mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader | xargs)
if [ "${compute_mode}" != "Exclusive_Process" ]; then
  echo "::error::expected compute mode Exclusive_Process, found ${compute_mode}"
  exit 1
fi

# Read memory, not --query-compute-apps: the container's PID namespace
# makes process queries come back empty even for a busy card.
MAX_USED_MIB=64
mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | xargs)
# A non-numeric reading ([N/A], empty) would make the comparison below
# error out, and an error inside `if` does not trip set -e: the guard
# would pass. Reject it here so an unreadable GPU fails closed.
case "${mem_used}" in
  ''|*[!0-9]*)
    echo "::error::could not read GPU memory use, got '${mem_used}'"
    exit 1
    ;;
esac
if [ "${mem_used}" -gt "${MAX_USED_MIB}" ]; then
  echo "::error::GPU is not idle: ${mem_used} MiB in use, limit ${MAX_USED_MIB} MiB"
  nvidia-smi
  exit 1
fi

TARGET_CLOCK_MHZ=1500
RETRIES=5
for i in $(seq 1 "${RETRIES}"); do
  gpu_clock=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits | xargs)
  echo "GPU clock: ${gpu_clock} MHz (attempt ${i}/${RETRIES})"
  if [ "${gpu_clock}" = "${TARGET_CLOCK_MHZ}" ]; then break; fi
  if [ "${i}" -eq "${RETRIES}" ]; then
    echo "::error::GPU frequency validation failed. Expected ${TARGET_CLOCK_MHZ} MHz, got ${gpu_clock} MHz."
    exit 1
  fi
  sleep 2
done
