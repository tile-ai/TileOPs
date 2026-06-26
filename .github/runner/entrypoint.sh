#!/usr/bin/env bash
set -euo pipefail

# Group-writable (0775 dirs / 0664 files) so compiled kernels in the shared /ci-cache are
# reusable across runner UIDs that share the ci-runner group.
umask 002

# Ad-hoc command passthrough: `docker run <image> python ...` / `bash -c ...` runs the
# given command directly (image verification, smoke tests; see README). With no args the
# container registers an ephemeral self-hosted runner (below).
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

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
# Make container stop signals reach the runner so an idle ephemeral runner shuts down promptly
# instead of waiting out docker stop's grace:
#   - RUNNER_MANUALLY_TRAP_SIG=1 makes run.sh trap SIGTERM/SIGINT and forward SIGINT to the
#     listener process group (its default mode does not trap, so the signal is otherwise lost).
#   - exec makes run.sh PID 1 so docker stop's SIGTERM is delivered to it; a bash wrapper as
#     PID 1 would ignore an untrapped SIGTERM and stall until SIGKILL.
# Trade-off: a stop signal cancels an in-progress job rather than draining it.
export RUNNER_MANUALLY_TRAP_SIG=1
exec ./run.sh
