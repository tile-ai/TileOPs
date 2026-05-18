#!/usr/bin/env bash
# reclaim_cache.sh — cache-reclaim primitives used by the
# .github/actions/reclaim-runner-disk composite action.
#
# The action.yml composite used to embed all reclaim logic inline, which
# meant the cache-trim paths could only be exercised on a self-hosted
# runner. That made it impossible to catch regressions at PR time; a bad
# trim rule only surfaced after merge when it had already corrupted the
# live cache. This script exposes the trim primitives as
# pytest-driveable subcommands so tests/test_reclaim_action.py can
# validate behaviour against a tmp_path fixture.
#
# The core safety invariant this script enforces is that some cache
# roots (notably /data7/shared/ci-cache/tilelang/autotuner) store
# *atomic* first-level subdirectories: consumers assume "directory
# exists => contents complete" and the directory is only valid if the
# sentinel file (best_config.json) is present. File-level `-mtime`
# trimming on these roots can delete the sentinel while leaving the
# rest of the directory in place, producing a half-dead state that
# crashes the next consumer. The atomic-trim and sentinel-repair
# subcommands here operate at whole-subdir granularity and never touch
# individual files inside an atomic root.
#
# Subcommands:
#
#   sentinel-repair <root> [<root>...]
#       For each atomic root, delete any first-level subdirectory that
#       does not contain ${SENTINEL_FILENAME:-best_config.json}. This
#       self-heals half-dead state left by older reclaim passes.
#
#   atomic-trim <age-days> <root> [<root>...]
#       For each atomic root, compute the newest file mtime anywhere
#       inside each first-level subdirectory. If that mtime is older
#       than <age-days> days, delete the whole subdirectory. Never
#       trims individual files inside atomic roots.
#
#   trim-files <age-days> <root> [<root>...]
#       File-level `-mtime +N -delete` trim for non-atomic cache
#       roots, followed by empty-directory cleanup. This is the legacy
#       behaviour used for /data7/shared/ci-cache/triton,
#       /data7/shared/ci-cache/pip, /data7/shared/ci-cache/wheels, etc.
#
# All subcommands are idempotent, tolerate missing roots (skipped
# silently), and never exit non-zero for per-directory errors so a
# transient `rm` failure cannot abort the whole reclaim step.

set -uo pipefail

SENTINEL_FILENAME="${SENTINEL_FILENAME:-best_config.json}"

_log() {
    echo "$@"
}

# sentinel_repair <root> [<root>...]
#
# Delete any first-level subdirectory of each root that is missing the
# sentinel file. Roots that do not exist are skipped. The sentinel
# filename is read from ${SENTINEL_FILENAME:-best_config.json}.
sentinel_repair() {
    local root subdir
    for root in "$@"; do
        [[ -z "$root" ]] && continue
        [[ -d "$root" ]] || continue
        while IFS= read -r -d '' subdir; do
            if [[ ! -e "${subdir}/${SENTINEL_FILENAME}" ]]; then
                _log "sentinel-repair: removing ${subdir} (missing ${SENTINEL_FILENAME})"
                rm -rf "$subdir" 2>/dev/null || true
            fi
        done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
}

# atomic_trim <age-days> <root> [<root>...]
#
# For each first-level subdirectory of each root, compute the newest
# file mtime in the whole subtree. If that mtime is older than
# <age-days> days, delete the whole subdirectory. Never deletes
# individual files; the unit of trim is the subdirectory.
atomic_trim() {
    local age_days="$1"
    shift
    local now_epoch cutoff root subdir newest_mtime
    # Validate age_days is a plain non-negative integer before feeding it
    # into arithmetic. Bash defaults to base-8 for literals with a leading
    # zero, so a caller-supplied value like "08"/"09" (e.g. from a
    # zero-padded workflow input) would abort the whole trim with
    # "value too great for base". We fail *open* — log and return 0 —
    # so a bad input can't crash the wider reclaim pass, and we force
    # base-10 parsing via `10#...` for defence in depth.
    if [[ ! "$age_days" =~ ^[0-9]+$ ]]; then
        _log "atomic-trim: ignoring invalid age_days='${age_days}' (expected non-negative integer)"
        return 0
    fi
    now_epoch=$(date +%s)
    cutoff=$(( now_epoch - 10#${age_days} * 86400 ))
    for root in "$@"; do
        [[ -z "$root" ]] && continue
        [[ -d "$root" ]] || continue
        while IFS= read -r -d '' subdir; do
            # Use the newest FILE mtime anywhere in the subtree — directory
            # mtimes must not participate. Rationale: a cache restore/extract
            # can bump the subdir's own mtime to "now" while every regular
            # file inside is still at its original (old) timestamp; counting
            # the dir mtime would wrongly mark the subdir as fresh and defeat
            # age-based reclaim. Only fall back to the subdir mtime when the
            # subtree has no regular files at all.
            newest_mtime=$(find "$subdir" -type f -printf '%T@\n' 2>/dev/null \
                | awk 'NR == 1 || $1 > max { max = $1 } END { if (NR > 0) printf "%d\n", max }')
            if [[ -z "$newest_mtime" ]]; then
                newest_mtime=$(stat -c %Y "$subdir" 2>/dev/null || echo 0)
            fi
            if (( newest_mtime < cutoff )); then
                _log "atomic-trim: removing ${subdir} (newest mtime ${newest_mtime} < cutoff ${cutoff})"
                rm -rf "$subdir" 2>/dev/null || true
            fi
        done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
    done
}

# trim_files <age-days> <root> [<root>...]
#
# File-level age-based trim for non-atomic caches. Deletes files older
# than <age-days>, then deletes any now-empty directories. Empty-dir
# cleanup does NOT gate on -mtime because file deletion bumps the
# parent directory's mtime to "now" and a mtime filter would leave the
# newly-empty dirs behind.
trim_files() {
    local age_days="$1"
    shift
    local root
    # Same validation as atomic_trim: refuse non-integer age_days and fail
    # open. `find -mtime` itself is tolerant of leading-zero strings, but
    # we keep the contract identical across subcommands so a caller that
    # passes a bogus value gets the same behaviour everywhere.
    if [[ ! "$age_days" =~ ^[0-9]+$ ]]; then
        _log "trim-files: ignoring invalid age_days='${age_days}' (expected non-negative integer)"
        return 0
    fi
    age_days=$(( 10#${age_days} ))
    for root in "$@"; do
        [[ -z "$root" ]] && continue
        [[ -d "$root" ]] || continue
        find "$root" -type f -mtime "+${age_days}" -delete 2>/dev/null || true
        find "$root" -depth -mindepth 1 -type d -empty -delete 2>/dev/null || true
    done
}

_usage() {
    cat >&2 <<'EOF'
usage: reclaim_cache.sh <subcommand> [args...]

subcommands:
  sentinel-repair <root> [<root>...]
  atomic-trim <age-days> <root> [<root>...]
  trim-files <age-days> <root> [<root>...]
EOF
    exit 2
}

main() {
    [[ $# -ge 1 ]] || _usage
    local cmd="$1"
    shift
    case "$cmd" in
        sentinel-repair)
            sentinel_repair "$@"
            ;;
        atomic-trim)
            [[ $# -ge 2 ]] || _usage
            atomic_trim "$@"
            ;;
        trim-files)
            [[ $# -ge 2 ]] || _usage
            trim_files "$@"
            ;;
        *)
            _usage
            ;;
    esac
}

# Allow sourcing the file in tests without executing main.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
