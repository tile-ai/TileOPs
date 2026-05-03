#!/usr/bin/env bash
# loop.sh <PR> — per-PR review orchestrator.
#
# One process per PR. Runs preflight once, classifies the PR once, then
# loops: poll for new commits / non-reviewer comments → run a Codex
# review round → update state → sleep. Terminates on APPROVE+stable,
# PR merged/closed, MAX_ROUNDS hit, or repeated Codex failure.
#
# Recommended invocation for unattended use:
#     nohup bash .claude/skills/review-tileops/loop.sh 1122 \
#         > review-pr-1122.log 2>&1 &
#
# State (under <task_root>/review/) is durable; killing or rebooting and
# re-invoking resumes from the last persisted round.
set -euo pipefail

PR="${1:?usage: loop.sh <PR_NUMBER>}"
[[ "$PR" =~ ^[0-9]+$ ]] || { echo "PR must be a positive integer" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="tile-ai/TileOPs"
MAX_ROUNDS=15
POLL_INTERVAL=180
CODEX_RETRY=3
# Stall safety: terminate after this many consecutive idle polls (no new
# commits / comments). MAX_IDLE * POLL_INTERVAL must comfortably exceed
# the longest legitimate in-progress counterpart round (codex review on a
# large PR, or a developer round of edit + test + push). 20 polls * 180s
# = 60min — wide enough that a single slow but live counterpart round
# does not trip terminate-stalled, narrow enough that a truly dead
# counterpart still exits within the hour. Wall-clock cap below caps
# absolute runaway risk.
MAX_IDLE=20
# Belt-and-suspenders: hard wall-clock cap independent of round/idle
# accounting, in case a logic bug skips both. Far above any realistic PR
# lifetime (hours), low enough that a runaway is bounded to a day.
MAX_WALL_CLOCK_HOURS=24
LOOP_START_EPOCH=$(date +%s)
# State root vs source root are distinct (Ibuki review on PR #1139):
#
# - REPO_PATH is the *state root* — main checkout. Used for `.foundry/
#   runs/`, fetches, refs, and managed worktrees so state is shared
#   across all worktrees of one repo. `git worktree list --porcelain 2>/dev/null`
#   always lists the main worktree first.
# - SOURCE_ROOT is the *source/policy root* — the repo that contains
#   this script. Used for `loading.yaml`, `criteria.md`, `procedure.md`,
#   and `.claude/review-checklists/` so the loop reads policy files
#   from the same branch that ships this script. Without the split,
#   a loop launched from a linked worktree would mix policy files
#   (criteria/procedure/loading from worktree branch via SKILL_DIR,
#   checklists from main branch via REPO_PATH).
REPO_PATH="$(git worktree list --porcelain 2>/dev/null | head -n 1 | sed 's/^worktree //')" \
  || { echo "loop.sh: not in a git repo" >&2; exit 1; }
SOURCE_ROOT="$(cd "$SKILL_DIR/../../.." && pwd)"

CRITERIA_PATH="$SKILL_DIR/criteria.md"
PROCEDURE_PATH="$SKILL_DIR/procedure.md"
LOADING_YAML="$SKILL_DIR/loading.yaml"
CHECKLISTS_DIR="$SOURCE_ROOT/.claude/review-checklists"

# Discover the local remote that points at tile-ai/TileOPs. Different clones
# use different remote names (origin in clones, upstream in forks, anything
# in CI environments) so we never hard-code.
TILEOPS_REMOTE=$(git -C "$REPO_PATH" remote -v \
  | awk '/tile-ai\/TileOPs(\.git)?[[:space:]]+\(fetch\)/ {print $1; exit}')
if [[ -z "$TILEOPS_REMOTE" ]]; then
  echo "loop.sh: no git remote in $REPO_PATH points at tile-ai/TileOPs" >&2
  exit 1
fi

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------------------
# Step A: preflight (once per loop start)
# ---------------------------------------------------------------------------
bash "$SKILL_DIR/preflight.sh" "$PR" || exit 1
export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"

# ---------------------------------------------------------------------------
# Step B: resolve task root + state dirs
# ---------------------------------------------------------------------------
# Resolve task root from PR body. Same convention as foundry pipeline:
# `Closes #N` → .foundry/runs/issue-N/, otherwise .foundry/runs/pr-<PR>/.
# Inlined to avoid a cross-repo dep on foundry just to run 10 lines of bash.
resolve_task_root() {
  local pr="$1" body issue
  body=$(gh pr view "$pr" --repo "$REPO" --json body --jq .body 2>/dev/null || echo "")
  issue=$(printf '%s' "$body" \
    | grep -oiE '(Closes|Fixes|Resolves)[[:space:]]+#[0-9]+' \
    | head -1 \
    | grep -oE '[0-9]+' \
    || true)
  if [[ -n "$issue" ]]; then
    printf '%s/.foundry/runs/issue-%s\n' "$REPO_PATH" "$issue"
  else
    printf '%s/.foundry/runs/pr-%s\n' "$REPO_PATH" "$pr"
  fi
}

TASK_ROOT=$(resolve_task_root "$PR")
RUN_DIR="$TASK_ROOT/review"
META="$RUN_DIR/meta.json"
CONTEXT="$RUN_DIR/context.json"
TASK_META="$TASK_ROOT/meta.json"
WORKTREE_DIR="$RUN_DIR/worktree"
# Per-PR ref outside refs/heads/ so it isn't a branch (no
# "checked-out-branch" rejection on subsequent fetches) and is namespaced
# per PR (no race against concurrent per-PR loops sharing FETCH_HEAD).
PR_REF="refs/review-loop/pr-$PR/head"
mkdir -p "$RUN_DIR/rounds" "$RUN_DIR/inbox-history"

# Maintain a dedicated worktree pinned at the PR's head sha. Codex reads
# source files from there; the user's main worktree is never disturbed.
# Worktree is on detached HEAD; advance each round via reset --hard.
sync_pr_worktree() {
  git -C "$REPO_PATH" fetch "$TILEOPS_REMOTE" "+pull/$PR/head:$PR_REF" \
    >/dev/null 2>&1 || {
      echo "loop.sh: failed to fetch pull/$PR/head from $TILEOPS_REMOTE" >&2
      return 1
    }
  local target
  target=$(git -C "$REPO_PATH" rev-parse "$PR_REF")
  if [[ ! -d "$WORKTREE_DIR/.git" && ! -e "$WORKTREE_DIR/.git" ]]; then
    git -C "$REPO_PATH" worktree add --detach "$WORKTREE_DIR" "$target" >/dev/null
  else
    git -C "$WORKTREE_DIR" reset --hard "$target" >/dev/null
  fi
}

# Drop the per-PR worktree and ref. Called on APPROVE convergence so the
# loop doesn't leave a full repo checkout sitting on disk per merged PR.
# Other state under $RUN_DIR (logs, prompts, retrospective) stays for
# postmortem inspection.
cleanup_pr_worktree() {
  if [[ -e "$WORKTREE_DIR" ]]; then
    git -C "$REPO_PATH" worktree remove --force "$WORKTREE_DIR" >/dev/null 2>&1 \
      || rm -rf "$WORKTREE_DIR"
  fi
  # Drop stale worktree admin entries left over if `worktree remove` failed
  # and we fell back to `rm -rf`. Without this, the next `worktree add` to
  # this path would fail with "already registered".
  git -C "$REPO_PATH" worktree prune >/dev/null 2>&1 || true
  git -C "$REPO_PATH" update-ref -d "$PR_REF" 2>/dev/null || true
}
sync_pr_worktree

# Task-level meta.json (cross-stage state, shared with future foundry-pipeline
# coexistence). Loop only writes the .stages.review.* fields; if foundry
# pipeline has already initialized the file, we leave its other stages alone.
ensure_task_meta() {
  [[ -f "$TASK_META" ]] && return 0
  mkdir -p "$TASK_ROOT"
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jq -n --arg now "$now" '{
    current_stage: "review",
    stages: {
      pipeline: { status: "pending", started_at: null, ended_at: null, rounds: 0, last_event: null },
      dev:      { status: "pending", started_at: null, ended_at: null, rounds: 0, last_event: null },
      review:   { status: "active",  started_at: $now, ended_at: null, rounds: 0, last_event: null }
    },
    updated_at: $now
  }' > "$TASK_META"
}

set_task_review_rounds() {
  ensure_task_meta
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jq --argjson n "$1" --arg now "$now" \
     '.stages.review.rounds = $n | .updated_at = $now' \
     "$TASK_META" > "$TASK_META.tmp" && mv "$TASK_META.tmp" "$TASK_META"
}

set_task_review_event() {
  ensure_task_meta
  local now
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  jq --arg ev "$1" --arg now "$now" \
     '.stages.review.last_event = $ev | .updated_at = $now' \
     "$TASK_META" > "$TASK_META.tmp" && mv "$TASK_META.tmp" "$TASK_META"
}

ensure_task_meta

# ---------------------------------------------------------------------------
# Step C: classify PR (once; persisted in context.json)
# ---------------------------------------------------------------------------
if [[ ! -f "$CONTEXT" ]]; then
  log "classifying PR #$PR …"
  PR_JSON=$(gh pr view "$PR" --repo "$REPO" --json title,headRefOid,author)
  TITLE=$(printf '%s' "$PR_JSON" | jq -r .title)
  AUTHOR=$(printf '%s' "$PR_JSON" | jq -r .author.login)

  TYPE=$(printf '%s' "$TITLE" | sed -nE 's/^\[([^]]+)\].*/\1/p')
  SCOPE=$(printf '%s' "$TITLE" | sed -nE 's/^\[[^]]+\]\[([^]]+)\].*/\1/p')

  # Resolve checklist names from loading.yaml via python (PyYAML).
  CHECKLISTS=$(python3 - "$LOADING_YAML" "$TYPE" "$SCOPE" <<'PY'
import sys, yaml
data = yaml.safe_load(open(sys.argv[1]))
selected = list(data.get("always", []))
for token in (sys.argv[2], sys.argv[3]):
    if not token:
        continue
    for cl in data.get("match", {}).get(token, []):
        if cl not in selected:
            selected.append(cl)
for cl in selected:
    print(cl)
PY
)

  jq -n \
    --arg title "$TITLE" \
    --arg author "$AUTHOR" \
    --arg type "$TYPE" \
    --arg scope "$SCOPE" \
    --argjson checklists "$(printf '%s\n' "$CHECKLISTS" | jq -R . | jq -s .)" \
    '{title:$title, author:$author, type:$type, scope:$scope, checklists:$checklists}' \
    > "$CONTEXT"

  log "classification: type=[$TYPE] scope=[$SCOPE] checklists=$(jq -r '.checklists|join(",")' "$CONTEXT")"
fi

# ---------------------------------------------------------------------------
# Step D: initialize loop-local meta
# ---------------------------------------------------------------------------
if [[ ! -f "$META" ]]; then
  jq -n --arg pr "$PR" --arg repo "$REPO" '{
    pr_number: ($pr|tonumber), repo: $repo,
    codex_session_id: null,
    round: 0,
    last_reviewed_sha: null,
    last_human_comment_id: 0,
    last_reviewer_comment_id: 0,
    last_codex_event: null,
    last_criteria_mtime: 0,
    consecutive_codex_failures: 0,
    consecutive_request_changes: 0,
    consecutive_idle: 0,
    status: "active"
  }' > "$META"
fi

REVIEWER_LOGIN=$(gh api user --jq .login)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
latest_human_comment_id() {
  # Max id of any non-reviewer comment (issue thread + PR review thread).
  local issue_max review_max
  issue_max=$(gh api "repos/$REPO/issues/$PR/comments" \
    --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\")|.id]|max // 0" \
    2>/dev/null || echo 0)
  review_max=$(gh api "repos/$REPO/pulls/$PR/comments" \
    --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\")|.id]|max // 0" \
    2>/dev/null || echo 0)
  echo $(( issue_max > review_max ? issue_max : review_max ))
}

set_meta_status() {
  local status="$1"
  jq --arg s "$status" '.status=$s' "$META" > "$META.tmp" && mv "$META.tmp" "$META"
}

# Compose the round-N prompt to <SNAP>.prompt.md.
compose_prompt() {
  local snap="$1"
  local n="$2"
  local include_criteria="$3"
  local include_procedure="$4"
  local diff_file="$5"
  local pr_state="$6"
  local head_sha="$7"
  local last_sha="$8"
  local last_event="$9"
  local inbox_block="${10}"
  local consecutive_rc="${11}"
  local out="$snap.prompt.md"

  {
    echo "# Review Loop — PR #$PR, Round $n / $MAX_ROUNDS"
    echo ""

    if [[ "$include_criteria" == "1" ]]; then
      echo "## Review output format"
      echo ""
      cat "$CRITERIA_PATH"
      echo ""
    fi

    if [[ "$include_procedure" == "1" ]]; then
      echo "## Review procedure"
      echo ""
      cat "$PROCEDURE_PATH"
      echo ""
    fi

    if [[ -n "$inbox_block" ]]; then
      echo "## Per-round guidance from human"
      echo ""
      echo "$inbox_block"
      echo ""
    fi

    if [[ "$n" -eq 1 ]]; then
      echo "## Project-specific regression guards"
      echo ""
      echo "Apply LAST, after the free-form review. These catch known regression classes the free-form pass can miss — they do not replace it."
      echo ""
      jq -r '.checklists[]' "$CONTEXT" | while read -r cl; do
        local path="$CHECKLISTS_DIR/$cl"
        if [[ -f "$path" ]]; then
          echo "### $cl"
          echo ""
          cat "$path"
          echo ""
        else
          echo "### $cl — MISSING (file not found at $path)"
          echo ""
        fi
      done
    fi

    if [[ "$consecutive_rc" -ge 3 ]]; then
      cat <<ANCHOR
## Divergence trigger — design re-anchoring (mandatory)

This PR has been REQUEST_CHANGES for $consecutive_rc rounds in a row. Stop iterating bottom-up. Re-anchor top-down.

1. **Re-read from disk** (not memory): \`docs/design/architecture.md\`, \`docs/design/ops-design.md\`, plus any design doc named by an active guard.
2. **Audit the recurring blocker.** One root design concern, or local patches on a moving target?
3. **Question your anchor.** Cite the design passage that grounds the blocker. No citation possible → you've over-fitted on trivial details.
4. **Decide:**
   - **Reaffirm** — cite the passage inline.
   - **Withdraw** — retract explicitly. Remaining unease becomes a summary question, not a blocker.
   - **Reframe** — restate once at the design level with citation; stop relitigating surface variants.

Required line at the top of the summary (before the trailer):
\`\`\`
Divergence introspection: <reaffirmed|withdrawn|reframed> — <one-line reason>
\`\`\`

If reaffirmed and the next round still doesn't converge, recommend human review in the summary.

ANCHOR
    fi

    echo "## PR state"
    echo "- HEAD sha: \`$head_sha\`"
    echo "- PR state: \`$pr_state\`"
    echo "- Previous reviewed sha: \`${last_sha:-(first round)}\`"
    echo "- Previous review event: \`${last_event:-(first round)}\`"
    echo ""

    echo "## Inputs (already prepared at the listed paths — do not re-fetch)"
    echo "- Diff: \`$diff_file\`"
    echo "- Unresolved review threads: \`$snap.unresolved-threads.json\`"
    echo "- New issue comments since last round: \`$snap.new-issue-comments.json\`"
    echo "- New review comments since last round: \`$snap.new-review-comments.json\`"
    echo "- CI status: \`$snap.ci.json\`"
    echo ""

    echo "## Task"
    echo ""
    if [[ "$n" -gt 1 ]]; then
      echo "Developer pushed / replied since the last round. Verify both:"
      echo ""
      echo "1. Prior blockers actually fixed — read the changed source, not the reply."
      echo "2. No new problems from the new commits."
      echo ""
      echo "Round 1's procedure and guards still apply (already in session memory)."
      echo ""
    fi
    echo "Run the procedure end to end and submit one atomic review."
    echo "**Free-form review (procedure step 2) is the primary review step**; project-specific guards apply LAST as a regression net, not in place of it."
    echo "The summary body MUST end with this trailer (the loop driver parses it; review is rejected without it):"
    echo ""
    echo '```'
    echo "<!-- review-loop: event=APPROVE|REQUEST_CHANGES; blockers=<N>; sha=$(printf '%s' "$head_sha" | cut -c1-7) -->"
    echo '```'
    echo ""
    echo "\`<N>\` = unresolved blockers (0 for APPROVE)."
  } > "$out"
}

# Run codex once with retries; output to <SNAP>.codex-last-message.txt.
run_codex_round() {
  local snap="$1"
  local prompt_file="$snap.prompt.md"
  local lastmsg="$snap.codex-last-message.txt"
  local events="$snap.codex-events.jsonl"
  local sid
  sid=$(jq -r .codex_session_id "$META")

  local attempt=0
  while (( attempt < CODEX_RETRY )); do
    if [[ "$sid" == "null" || -z "$sid" ]]; then
      codex --dangerously-bypass-approvals-and-sandbox exec \
        --json --output-last-message "$lastmsg" --cd "$WORKTREE_DIR" \
        "$(cat "$prompt_file")" > "$events" 2>&1 || true
    else
      # `codex exec resume` does not accept --cd; the session already
      # remembers its cwd from the initial `exec`, but cd anyway as a
      # belt-and-suspenders for source-file lookups.
      ( cd "$WORKTREE_DIR" && \
        codex --dangerously-bypass-approvals-and-sandbox exec resume "$sid" \
          --json --output-last-message "$lastmsg" \
          "$(cat "$prompt_file")" ) > "$events" 2>&1 || true
    fi

    if [[ -s "$lastmsg" ]]; then
      # Capture session id after first successful round.
      if [[ "$sid" == "null" || -z "$sid" ]]; then
        sid=$(tail -n 50 ~/.codex/session_index.jsonl 2>/dev/null \
          | jq -rs 'sort_by(.updated_at)|last|.id' 2>/dev/null || echo "")
        [[ -n "$sid" && "$sid" != "null" ]] && \
          jq --arg s "$sid" '.codex_session_id=$s' "$META" \
            > "$META.tmp" && mv "$META.tmp" "$META"
      fi
      return 0
    fi

    attempt=$((attempt+1))
    log "codex attempt $attempt empty/failed; retrying in 10s …"
    # Surface the tail of the events file so the human log shows *why*
    # — otherwise repeated failures look identical and the cause stays
    # buried in $events.
    tail -n 5 "$events" 2>/dev/null | sed 's/^/  | /' >&2 || true
    sleep 10
  done

  log "codex failed $CODEX_RETRY times — stopping"
  set_meta_status "error"
  return 1
}

# State of the reviewer's most recent review on the PR — APPROVED,
# CHANGES_REQUESTED, COMMENTED, or DISMISSED (returned by GitHub when a
# stale-approval rule retracts an APPROVE on new commits). Returns
# "NONE" if there's no review yet or the call fails.
latest_reviewer_review_state() {
  gh api "repos/$REPO/pulls/$PR/reviews" \
    --jq "[.[]|select(.user.login==\"$REVIEWER_LOGIN\")]|sort_by(.submitted_at)|last|.state // \"NONE\"" \
    2>/dev/null || echo NONE
}

# Final convergence path: introspection + retrospective + worktree cleanup, exit 0.
converge_and_exit() {
  log "converged — APPROVE on PR #$PR. running introspection + retrospective"
  set_meta_status "success"
  set_task_review_event APPROVE
  run_introspection
  run_retrospective
  log "checklist-suggestions: $RUN_DIR/checklist-suggestions.md"
  log "retrospective:        $RUN_DIR/retrospective.md"
  cleanup_pr_worktree
  log "worktree removed: $WORKTREE_DIR"
  exit 0
}

# Parse the trailer from the latest reviewer-login review on the PR.
parse_trailer() {
  local body trailer
  body=$(gh api "repos/$REPO/pulls/$PR/reviews" \
    --jq "[.[]|select(.user.login==\"$REVIEWER_LOGIN\")]|sort_by(.submitted_at)|last|.body" \
    2>/dev/null || echo "")
  trailer=$(printf '%s' "$body" \
    | grep -oE '<!-- review-loop: event=(APPROVE|REQUEST_CHANGES); blockers=[0-9]+; sha=[a-f0-9]{7} -->' \
    | tail -1)
  printf '%s' "$trailer"
}

# Run a one-off Codex prompt to produce a small artifact (introspection/retro).
run_codex_oneoff() {
  local prompt="$1" outfile="$2" sid
  sid=$(jq -r .codex_session_id "$META")
  if [[ "$sid" == "null" || -z "$sid" ]]; then
    return 1
  fi
  ( cd "$WORKTREE_DIR" && \
    codex --dangerously-bypass-approvals-and-sandbox exec resume "$sid" \
      --output-last-message "$outfile" \
      "$prompt" ) >/dev/null 2>&1 || true
  [[ -s "$outfile" ]]
}

run_introspection() {
  local outfile="$RUN_DIR/checklist-suggestions.md"
  local prompt='Look back across all rounds of this PR (including findings the developer fixed). Propose at most THREE bullets for `.claude/review-checklists/` that, in hindsight, would have caught a finding faster.

Output ONLY (no preamble, no closing):

## Proposed checklist additions

- **Target file**: `.claude/review-checklists/<file>.md`
  **Proposed bullet**: (one line, bold lead phrase)
  **Why this PR caught it**: (one line, round + finding)

Each proposal must be:
- **Executable** — pass/fail decidable without further interpretation.
- **Lean** — one sentence, no rationale, no examples.
- **Not implementation-coupled** — survives 5 future implementations (no file paths, function names, refactor-prone class names).

If nothing qualifies, output exactly: No proposed additions.'
  run_codex_oneoff "$prompt" "$outfile" || true
}

run_retrospective() {
  local outfile="$RUN_DIR/retrospective.md"
  local prompt='Loop terminating. Output ONLY a terse retrospective in markdown — no preamble, no closing remarks.

Required:
1. **Problem** — 1–2 lines: core issue with this PR.
2. **Resolution** — `resolved` | `partial` | `unresolved`, one line on what is done.

Optional (omit if not substantive):
3. **Technical approach** — 1–2 lines on techniques applied.
4. **Follow-up** — concrete next-step items, one per line.

Action-oriented. No long prose.'
  run_codex_oneoff "$prompt" "$outfile" || true
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
log "loop started — PR #$PR, MAX_ROUNDS=$MAX_ROUNDS, POLL_INTERVAL=${POLL_INTERVAL}s, task_root=$TASK_ROOT"

while true; do
  ROUND=$(jq -r .round "$META")

  # Wall-clock backstop: independent of round/idle accounting, in case a
  # logic bug skips both. Runs before any GitHub API call.
  WALL_CLOCK_S=$(( $(date +%s) - LOOP_START_EPOCH ))
  if (( WALL_CLOCK_S > MAX_WALL_CLOCK_HOURS * 3600 )); then
    log "wall-clock cap (${MAX_WALL_CLOCK_HOURS}h) exceeded — runaway, exiting"
    set_meta_status "runaway"
    exit 6
  fi

  # External / hard-cutoff terminations come first.
  PR_VIEW=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid,isDraft 2>/dev/null) \
    || { log "gh pr view failed; sleeping ${POLL_INTERVAL}s"; sleep "$POLL_INTERVAL"; continue; }
  PR_STATE=$(printf '%s' "$PR_VIEW" | jq -r .state)
  HEAD_SHA=$(printf '%s' "$PR_VIEW" | jq -r .headRefOid)

  if [[ "$PR_STATE" == "MERGED" || "$PR_STATE" == "CLOSED" ]]; then
    log "PR is $PR_STATE — exiting"
    set_meta_status "external"
    exit 2
  fi

  if [[ "$ROUND" -ge "$MAX_ROUNDS" ]]; then
    log "reached MAX_ROUNDS ($MAX_ROUNDS) without APPROVE — diverged, human attention needed"
    set_meta_status "diverged"
    exit 3
  fi

  LAST_SHA=$(jq -r .last_reviewed_sha "$META")
  LAST_EVENT=$(jq -r .last_codex_event "$META")
  LAST_HUMAN_ID_PREV=$(jq -r .last_human_comment_id "$META")
  LATEST_HUMAN_ID=$(latest_human_comment_id)

  # Resume / restart: if local meta says we approved last time, converge
  # only if GitHub still shows APPROVED *and* nothing has changed since
  # the approval. New commits auto-dismiss approvals (state goes
  # DISMISSED), but new issue/review comments do not — so a comments-
  # changed check is still required to avoid silently skipping unread
  # human feedback. Any miss → fall through to a fresh round on the
  # current head.
  if [[ "$LAST_EVENT" == "APPROVE" ]]; then
    GH_REVIEW_STATE=$(latest_reviewer_review_state)
    if [[ "$GH_REVIEW_STATE" == "APPROVED" \
          && "$HEAD_SHA" == "$LAST_SHA" \
          && "$LATEST_HUMAN_ID" == "$LAST_HUMAN_ID_PREV" ]]; then
      converge_and_exit
    fi
    log "prior APPROVE no longer current (gh=$GH_REVIEW_STATE, head_changed=$([[ "$HEAD_SHA" != "$LAST_SHA" ]] && echo y || echo n), comments_changed=$([[ "$LATEST_HUMAN_ID" != "$LAST_HUMAN_ID_PREV" ]] && echo y || echo n)) — re-reviewing current head"
    jq '.last_codex_event="DISMISSED" | .last_reviewed_sha=null' \
      "$META" > "$META.tmp" && mv "$META.tmp" "$META"
    LAST_EVENT="DISMISSED"
    LAST_SHA="null"
  fi

  # Anything new since last round?
  if [[ "$ROUND" -gt 0 \
        && "$HEAD_SHA" == "$LAST_SHA" \
        && "$LATEST_HUMAN_ID" == "$LAST_HUMAN_ID_PREV" ]]; then
    CONSECUTIVE_IDLE=$(jq -r '.consecutive_idle // 0' "$META")
    NEW_IDLE=$((CONSECUTIVE_IDLE + 1))
    jq --argjson n "$NEW_IDLE" '.consecutive_idle=$n' "$META" \
      > "$META.tmp" && mv "$META.tmp" "$META"
    if (( NEW_IDLE >= MAX_IDLE )); then
      log "idle for $NEW_IDLE consecutive polls (MAX_IDLE=$MAX_IDLE) — stalled, exiting"
      set_meta_status "stalled"
      exit 5
    fi
    log "no new commits / comments — idle $NEW_IDLE/$MAX_IDLE, sleeping ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
    continue
  fi

  # ---- Run a review round ----
  NEXT_ROUND=$((ROUND + 1))
  N=$(printf '%02d' "$NEXT_ROUND")
  SNAP="$RUN_DIR/rounds/round-$N"
  log "round $NEXT_ROUND — gathering inputs (head=${HEAD_SHA:0:7})"

  # Move the worktree forward to this round's HEAD so Codex reads source from
  # the actual reviewed sha (not the user's main worktree branch).
  sync_pr_worktree

  # Diff
  if [[ "$LAST_SHA" == "null" || -z "$LAST_SHA" ]]; then
    gh pr diff "$PR" --repo "$REPO" > "$SNAP.full.diff"
    DIFF_FILE="$SNAP.full.diff"
  else
    git -C "$REPO_PATH" diff "$LAST_SHA".."$HEAD_SHA" > "$SNAP.incremental.diff"
    DIFF_FILE="$SNAP.incremental.diff"
  fi

  # Unresolved threads
  gh api graphql -f query='
    query($owner:String!,$repo:String!,$pr:Int!){
      repository(owner:$owner,name:$repo){
        pullRequest(number:$pr){
          reviewThreads(first:100){
            nodes{ isResolved path comments(first:5){ nodes{ author{login} body } } }
          }
        }
      }
    }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
    --jq '.data.repository.pullRequest.reviewThreads.nodes|map(select(.isResolved==false))' \
    > "$SNAP.unresolved-threads.json" 2>/dev/null || echo '[]' > "$SNAP.unresolved-threads.json"

  # New comments since last round
  gh api "repos/$REPO/issues/$PR/comments" \
    --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\" and .id>$LAST_HUMAN_ID_PREV)|{id,user:.user.login,body,created_at}]" \
    > "$SNAP.new-issue-comments.json" 2>/dev/null || echo '[]' > "$SNAP.new-issue-comments.json"
  gh api "repos/$REPO/pulls/$PR/comments" \
    --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\" and .id>$LAST_HUMAN_ID_PREV)|{id,user:.user.login,path,line,body,created_at}]" \
    > "$SNAP.new-review-comments.json" 2>/dev/null || echo '[]' > "$SNAP.new-review-comments.json"

  # CI
  gh pr checks "$PR" --repo "$REPO" --json name,state,conclusion \
    > "$SNAP.ci.json" 2>/dev/null || echo '[]' > "$SNAP.ci.json"

  # Inbox
  INBOX_BLOCK=""
  if [[ -s "$RUN_DIR/inbox.md" ]]; then
    INBOX_BLOCK=$(cat "$RUN_DIR/inbox.md")
    mv "$RUN_DIR/inbox.md" "$RUN_DIR/inbox-history/round-$N.md"
    : > "$RUN_DIR/inbox.md"
  fi

  # procedure.md is round-1-only: round 2+ replaces it with the short
  # "iteration round" block in compose_prompt. criteria.md uses an mtime
  # gate — re-inline only when the format spec changed since last
  # inclusion (Codex's session memory carries unchanged content forward).
  CRITERIA_MTIME=$(stat -c %Y "$CRITERIA_PATH")
  LAST_CRITERIA_MTIME=$(jq -r '.last_criteria_mtime // 0' "$META")
  INCLUDE_CRITERIA=0
  INCLUDE_PROCEDURE=0
  if [[ "$NEXT_ROUND" -eq 1 || "$CRITERIA_MTIME" -gt "$LAST_CRITERIA_MTIME" ]]; then
    INCLUDE_CRITERIA=1
  fi
  if [[ "$NEXT_ROUND" -eq 1 ]]; then
    INCLUDE_PROCEDURE=1
  fi

  CONSECUTIVE_RC=$(jq -r '.consecutive_request_changes // 0' "$META")
  compose_prompt "$SNAP" "$NEXT_ROUND" "$INCLUDE_CRITERIA" "$INCLUDE_PROCEDURE" \
    "$DIFF_FILE" "$PR_STATE" "$HEAD_SHA" "$LAST_SHA" "$LAST_EVENT" "$INBOX_BLOCK" \
    "$CONSECUTIVE_RC"

  # Convergence Rule 1 — same-SHA APPROVE skip. If a prior round in
  # this loop run already APPROVED the current HEAD, reuse that
  # outcome verbatim and skip codex (deterministic, file-only).
  RULE1_DECISION=$(RUN_DIR="$RUN_DIR" NEXT_ROUND="$NEXT_ROUND" HEAD_SHA="$HEAD_SHA" \
    LATEST_HUMAN_ID="$LATEST_HUMAN_ID" \
    bash "$SKILL_DIR/round-pre.sh" || echo "proceed")
  if [[ "$RULE1_DECISION" == "skip" ]]; then
    log "round $NEXT_ROUND — Rule 1: prior APPROVE on $HEAD_SHA found; skipping codex"
    EVENT="APPROVE"
    BLOCKERS=$(jq -r '.blockers_after // 0' "$SNAP.json")
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    jq --argjson r "$NEXT_ROUND" --arg sha "$HEAD_SHA" --arg now "$NOW" \
       --argjson hid "$LATEST_HUMAN_ID" --arg ev "$EVENT" \
       --argjson cm "$CRITERIA_MTIME" \
       '.round=$r | .last_reviewed_sha=$sha | .last_human_comment_id=$hid
        | .last_codex_event=$ev | .last_criteria_mtime=$cm
        | .consecutive_request_changes=0
        | .consecutive_codex_failures=0
        | .consecutive_idle=0
        | .last_reviewed_at=$now' \
       "$META" > "$META.tmp" && mv "$META.tmp" "$META"
    set_task_review_rounds "$NEXT_ROUND"
    set_task_review_event "$EVENT"
    log "round $NEXT_ROUND done (codex skipped) — event=$EVENT blockers=$BLOCKERS sha=${HEAD_SHA:0:7}"
    if [[ "$EVENT" == "APPROVE" ]]; then
      POST_HEAD_SHA=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
      POST_HUMAN_ID=$(latest_human_comment_id)
      if [[ "$POST_HEAD_SHA" == "$HEAD_SHA" && "$POST_HUMAN_ID" == "$LATEST_HUMAN_ID" ]]; then
        converge_and_exit
      fi
    fi
    sleep "$POLL_INTERVAL"
    continue
  fi

  log "round $NEXT_ROUND — invoking codex"
  if ! run_codex_round "$SNAP"; then
    exit 1
  fi

  TRAILER=$(parse_trailer)
  if [[ -z "$TRAILER" ]]; then
    log "round $NEXT_ROUND — codex output missing trailer; treating as error"
    set_meta_status "error"
    exit 1
  fi

  EVENT=$(printf '%s' "$TRAILER" | sed -nE 's/.*event=([A-Z_]+).*/\1/p')
  BLOCKERS=$(printf '%s' "$TRAILER" | sed -nE 's/.*blockers=([0-9]+).*/\1/p')

  NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if [[ "$EVENT" == "REQUEST_CHANGES" ]]; then
    NEW_RC=$((CONSECUTIVE_RC + 1))
  else
    NEW_RC=0
  fi
  jq --argjson r "$NEXT_ROUND" --arg sha "$HEAD_SHA" --arg now "$NOW" \
     --argjson hid "$LATEST_HUMAN_ID" --arg ev "$EVENT" \
     --argjson cm "$CRITERIA_MTIME" --argjson rc "$NEW_RC" \
     '.round=$r | .last_reviewed_sha=$sha | .last_human_comment_id=$hid
      | .last_codex_event=$ev | .last_criteria_mtime=$cm
      | .consecutive_request_changes=$rc
      | .consecutive_codex_failures=0
      | .consecutive_idle=0
      | .last_reviewed_at=$now' \
     "$META" > "$META.tmp" && mv "$META.tmp" "$META"

  jq -n --argjson r "$NEXT_ROUND" --arg now "$NOW" \
    --arg sha_b "${LAST_SHA:-null}" --arg sha_a "$HEAD_SHA" \
    --arg ev "$EVENT" --argjson bl "$BLOCKERS" \
    --argjson hid "$LATEST_HUMAN_ID" \
    '{round:$r, finished_at:$now, head_sha_before:$sha_b, head_sha_after:$sha_a,
      codex_event:$ev, blockers_after:$bl,
      last_human_comment_id:$hid}' > "$SNAP.json"

  set_task_review_rounds "$NEXT_ROUND"
  set_task_review_event "$EVENT"

  # Convergence Rule 2 — same-path 3-strike monitor. Updates
  # region-history.json and labels the PR `agent-stuck` once any
  # path has produced a blocker for >= 3 consecutive rounds.
  # Monitoring-only: must not mutate blockers, threads, or PR title.
  #
  # Source artifact contract (bug-2 fix): only inline comments tied to
  # codex's *latest* review count as blockers, and only when that
  # review's top-level state is REQUEST_CHANGES. This is the most
  # reliable native signal that codex itself flagged the comment as a
  # blocker — non-blocker review states (APPROVE, COMMENT) emit zero
  # blocker paths so nits, suggestions, and "no-action" remarks never
  # advance the agent-stuck counter.
  #
  # Algorithm:
  #   1. Fetch the latest REVIEWER_LOGIN review by submitted_at.
  #   2. If state == REQUEST_CHANGES → fetch its inline comments via
  #      /reviews/<id>/comments and extract path/line/id.
  #   3. Otherwise (APPROVE / COMMENT / DISMISSED / NONE) → write [].
  # Best-effort: any gh failure degrades to an empty list so Rule 2
  # stays monitor-only.
  CODEX_BLOCKERS_JSON="$SNAP.codex-blockers.json"
  LAST_REVIEWER_COMMENT_ID=$(jq -r '.last_reviewer_comment_id // 0' "$META")
  LATEST_REVIEW_JSON=$(gh api "repos/$REPO/pulls/$PR/reviews" \
    --jq "[.[]|select(.user.login==\"$REVIEWER_LOGIN\")]|sort_by(.submitted_at)|last // {}" \
    2>/dev/null || echo '{}')
  LATEST_REVIEW_STATE=$(printf '%s' "$LATEST_REVIEW_JSON" \
    | jq -r '.state // "NONE"' 2>/dev/null || echo NONE)
  LATEST_REVIEW_ID=$(printf '%s' "$LATEST_REVIEW_JSON" \
    | jq -r '.id // empty' 2>/dev/null || echo "")
  if [[ "$LATEST_REVIEW_STATE" == "CHANGES_REQUESTED" \
        || "$LATEST_REVIEW_STATE" == "REQUEST_CHANGES" ]] \
        && [[ -n "$LATEST_REVIEW_ID" ]]; then
    gh api "repos/$REPO/pulls/$PR/reviews/$LATEST_REVIEW_ID/comments" --paginate \
      --jq "[.[]|{id,user:.user.login,path,line,body,created_at}]" \
      > "$CODEX_BLOCKERS_JSON" 2>/dev/null || echo '[]' > "$CODEX_BLOCKERS_JSON"
  else
    echo '[]' > "$CODEX_BLOCKERS_JSON"
  fi
  # Advance the reviewer-comment watermark so the next round only sees
  # comments newly posted by codex.
  NEW_REVIEWER_MAX=$(jq -r '[.[].id // 0]|max // 0' "$CODEX_BLOCKERS_JSON" 2>/dev/null || echo 0)
  if [[ "$NEW_REVIEWER_MAX" -gt "$LAST_REVIEWER_COMMENT_ID" ]]; then
    jq --argjson n "$NEW_REVIEWER_MAX" '.last_reviewer_comment_id=$n' \
      "$META" > "$META.tmp" && mv "$META.tmp" "$META"
  fi

  RUN_DIR="$RUN_DIR" ROUND="$NEXT_ROUND" \
    COMMENTS_JSON="$CODEX_BLOCKERS_JSON" \
    REPO="$REPO" PR="$PR" \
    bash "$SKILL_DIR/round-post.sh" || \
      log "round-post.sh: non-fatal failure (monitor-only); continuing"

  log "round $NEXT_ROUND done — event=$EVENT blockers=$BLOCKERS sha=${HEAD_SHA:0:7}"

  # APPROVE this round → exit, but re-check stability first. HEAD_SHA
  # and LATEST_HUMAN_ID were snapshotted before Codex ran (potentially
  # minutes ago); a push or non-reviewer comment arriving during the
  # review must not be silently skipped. If anything moved, fall
  # through to sleep + next iteration, which will pick up the change.
  if [[ "$EVENT" == "APPROVE" ]]; then
    POST_HEAD_SHA=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)
    POST_HUMAN_ID=$(latest_human_comment_id)
    if [[ "$POST_HEAD_SHA" == "$HEAD_SHA" && "$POST_HUMAN_ID" == "$LATEST_HUMAN_ID" ]]; then
      converge_and_exit
    fi
    log "APPROVE produced but state moved during review (head_changed=$([[ "$POST_HEAD_SHA" != "$HEAD_SHA" ]] && echo y || echo n), comments_changed=$([[ "$POST_HUMAN_ID" != "$LATEST_HUMAN_ID" ]] && echo y || echo n)) — falling through"
  fi

  sleep "$POLL_INTERVAL"
done
