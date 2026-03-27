#!/usr/bin/env bash
set -euo pipefail

# ── Validate required environment variables ──────────────────────────────────
: "${RUNNER_TOKEN:?Environment variable RUNNER_TOKEN is required}"
: "${RUNNER_URL:?Environment variable RUNNER_URL is required}"

RUNNER_NAME="${RUNNER_NAME:-$(hostname)}"
RUNNER_LABELS="${RUNNER_LABELS:-self-hosted,tile-ops,venv}"
RUNNER_WORKDIR="${RUNNER_WORKDIR:-_work}"

# ── Cleanup function — deregister runner on exit ─────────────────────────────
cleanup() {
    echo "Removing runner registration..."
    ./config.sh remove --token "${RUNNER_TOKEN}" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# ── Configure the runner (ephemeral: one job per lifecycle) ──────────────────
./config.sh \
    --url "${RUNNER_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME}" \
    --labels "${RUNNER_LABELS}" \
    --work "${RUNNER_WORKDIR}" \
    --ephemeral \
    --replace \
    --unattended \
    --disableupdate

# ── Run one job, then exit ───────────────────────────────────────────────────
./run.sh
