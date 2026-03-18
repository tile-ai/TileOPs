#!/bin/bash
# Wait for GPU 0 to become idle before running benchmarks.
#
# Exit codes:
#   0 — GPU 0 is idle (no compute processes)
#   1 — nvidia-smi not found (no GPU)
#   2 — GPU 0 still busy after all retries
#
# Usage: wait-for-gpu.sh [max_retries] [interval_seconds]

set -euo pipefail

MAX_RETRIES="${1:-5}"
INTERVAL="${2:-30}"

if ! command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi not found — no GPU available"
  exit 1
fi

for ((i = 1; i <= MAX_RETRIES; i++)); do
  PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null || true)
  if [[ -z "$PROCS" ]]; then
    echo "GPU 0 is idle"
    exit 0
  fi

  if [[ "$i" -lt "$MAX_RETRIES" ]]; then
    echo "GPU 0 busy (attempt $i/$MAX_RETRIES), waiting ${INTERVAL}s..."
    sleep "$INTERVAL"
  fi
done

echo "GPU 0 still busy after $MAX_RETRIES attempts"
exit 2
