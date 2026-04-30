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
MAX_ROUNDS=10
POLL_INTERVAL=180
CODEX_RETRY=3
REPO_PATH="$(git rev-parse --show-toplevel)"

CRITERIA_PATH="$SKILL_DIR/criteria.md"
PROCEDURE_PATH="$SKILL_DIR/procedure.md"
LOADING_YAML="$SKILL_DIR/loading.yaml"
CHECKLISTS_DIR="$REPO_PATH/.claude/review-checklists"

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
    last_codex_event: null,
    last_criteria_mtime: 0,
    last_procedure_mtime: 0,
    consecutive_codex_failures: 0,
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
  local out="$snap.prompt.md"

  {
    echo "# Review Loop — PR #$PR, Round $n / $MAX_ROUNDS"
    echo ""

    if [[ "$include_criteria" == "1" ]]; then
      echo "## Review output format (criteria.md)"
      echo ""
      cat "$CRITERIA_PATH"
      echo ""
    fi

    if [[ "$include_procedure" == "1" ]]; then
      echo "## Review procedure (procedure.md)"
      echo ""
      cat "$PROCEDURE_PATH"
      echo ""
    fi

    if [[ -n "$inbox_block" ]]; then
      echo "## Per-round guidance from human (inbox, one-shot)"
      echo ""
      echo "$inbox_block"
      echo ""
    fi

    if [[ "$n" -eq 1 ]]; then
      local title type scope
      title=$(jq -r .title "$CONTEXT")
      type=$(jq -r .type "$CONTEXT")
      scope=$(jq -r .scope "$CONTEXT")
      echo "## PR classification"
      echo ""
      echo "- Title: \`$title\`"
      echo "- Type: \`[$type]\`   Scope: \`[$scope]\`"
      echo "- Loaded checklists:"
      jq -r '.checklists[]' "$CONTEXT" | while read -r cl; do
        echo "  - \`.claude/review-checklists/$cl\`"
      done
      echo ""
      echo "## Checklist content"
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
    else
      echo "## Iteration round (round $n of $MAX_ROUNDS)"
      echo ""
      echo "Developer pushed changes since the last review. Verify:"
      echo ""
      echo "1. Are the prior blockers actually resolved?"
      echo "2. Were any new problems introduced by the new commits?"
      echo ""
    fi

    if [[ "$n" -ge 7 ]]; then
      cat <<'ANCHOR'
## Round 7+ — Design re-anchoring (mandatory)

The bottom-up checklist is failing to converge. Re-anchor top-down.

1. **Re-read from disk** (not memory): `docs/design/architecture.md`, `docs/design/ops-design.md`, plus any design doc named by your active checklist items.
2. **Audit the blocker thread.** One root concern, or local patches on a moving target?
3. **Question your anchor.** Cite the design passage that grounds the blocker. No citation possible → overfitted.
4. **Decide:**
   - **Reaffirm** — cite the passage inline.
   - **Withdraw** — retract explicitly. Remaining unease becomes a summary question, not a blocker.
   - **Reframe** — restate once at the design level with citation; stop relitigating surface variants.

Required line at the top of the summary (before the trailer):
```
Round-7 introspection: <reaffirmed|withdrawn|reframed> — <one-line reason>
```

Stop bickering with the developer over minor details. If `reaffirmed` and the author shows no movement for 2+ further rounds, state in the summary that the PR is stalling and recommend human review.

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
    echo "Read the diff at the path above and any changed source files referenced therein, in full. Apply the loaded checklists. Submit ONE atomic review on \`$REPO\` PR #$PR via \`gh\` per the format spec."
    echo ""
    echo "The summary body MUST end with this trailer line (the loop driver parses it; review is rejected without it):"
    echo ""
    echo '```'
    echo "<!-- review-loop: event=APPROVE|REQUEST_CHANGES; blockers=<N>; sha=$(printf '%s' "$head_sha" | cut -c1-7) -->"
    echo '```'
    echo ""
    echo "\`<N>\` = unresolved blockers after this review (0 for APPROVE)."
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
      codex --dangerously-bypass-approvals-and-sandbox exec resume "$sid" \
        --json --output-last-message "$lastmsg" --cd "$WORKTREE_DIR" \
        "$(cat "$prompt_file")" > "$events" 2>&1 || true
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
    sleep 10
  done

  log "codex failed $CODEX_RETRY times — stopping"
  set_meta_status "error"
  return 1
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
  codex --dangerously-bypass-approvals-and-sandbox exec resume "$sid" \
    --output-last-message "$outfile" --cd "$WORKTREE_DIR" \
    "$prompt" >/dev/null 2>&1 || true
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

  # Convergence: prior round APPROVE, head sha unchanged, no new comments.
  if [[ "$LAST_EVENT" == "APPROVE" \
        && "$HEAD_SHA" == "$LAST_SHA" \
        && "$LATEST_HUMAN_ID" == "$LAST_HUMAN_ID_PREV" ]]; then
    log "converged — APPROVE on stable sha, no new comments. running introspection + retrospective"
    set_meta_status "success"
    set_task_review_event APPROVE
    run_introspection
    run_retrospective
    log "checklist-suggestions: $RUN_DIR/checklist-suggestions.md"
    log "retrospective:        $RUN_DIR/retrospective.md"
    cleanup_pr_worktree
    log "worktree removed: $WORKTREE_DIR"
    exit 0
  fi

  # Anything new since last round?
  if [[ "$ROUND" -gt 0 \
        && "$HEAD_SHA" == "$LAST_SHA" \
        && "$LATEST_HUMAN_ID" == "$LAST_HUMAN_ID_PREV" ]]; then
    log "no new commits / comments — sleeping ${POLL_INTERVAL}s"
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

  # criteria.md / procedure.md mtime checks — round 1 always inlines both,
  # later rounds only re-inline when the file changed since last inclusion
  # (Codex's session memory carries unchanged content forward).
  CRITERIA_MTIME=$(stat -c %Y "$CRITERIA_PATH")
  PROCEDURE_MTIME=$(stat -c %Y "$PROCEDURE_PATH")
  LAST_CRITERIA_MTIME=$(jq -r '.last_criteria_mtime // 0' "$META")
  LAST_PROCEDURE_MTIME=$(jq -r '.last_procedure_mtime // 0' "$META")
  INCLUDE_CRITERIA=0
  INCLUDE_PROCEDURE=0
  if [[ "$NEXT_ROUND" -eq 1 || "$CRITERIA_MTIME" -gt "$LAST_CRITERIA_MTIME" ]]; then
    INCLUDE_CRITERIA=1
  fi
  if [[ "$NEXT_ROUND" -eq 1 || "$PROCEDURE_MTIME" -gt "$LAST_PROCEDURE_MTIME" ]]; then
    INCLUDE_PROCEDURE=1
  fi

  compose_prompt "$SNAP" "$NEXT_ROUND" "$INCLUDE_CRITERIA" "$INCLUDE_PROCEDURE" \
    "$DIFF_FILE" "$PR_STATE" "$HEAD_SHA" "$LAST_SHA" "$LAST_EVENT" "$INBOX_BLOCK"

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
  jq --argjson r "$NEXT_ROUND" --arg sha "$HEAD_SHA" --arg now "$NOW" \
     --argjson hid "$LATEST_HUMAN_ID" --arg ev "$EVENT" \
     --argjson cm "$CRITERIA_MTIME" --argjson pm "$PROCEDURE_MTIME" \
     '.round=$r | .last_reviewed_sha=$sha | .last_human_comment_id=$hid
      | .last_codex_event=$ev | .last_criteria_mtime=$cm
      | .last_procedure_mtime=$pm
      | .consecutive_codex_failures=0 | .last_reviewed_at=$now' \
     "$META" > "$META.tmp" && mv "$META.tmp" "$META"

  jq -n --argjson r "$NEXT_ROUND" --arg now "$NOW" \
    --arg sha_b "${LAST_SHA:-null}" --arg sha_a "$HEAD_SHA" \
    --arg ev "$EVENT" --argjson bl "$BLOCKERS" \
    '{round:$r, finished_at:$now, head_sha_before:$sha_b, head_sha_after:$sha_a,
      codex_event:$ev, blockers_after:$bl}' > "$SNAP.json"

  set_task_review_rounds "$NEXT_ROUND"
  set_task_review_event "$EVENT"

  log "round $NEXT_ROUND done — event=$EVENT blockers=$BLOCKERS sha=${HEAD_SHA:0:7}"

  # Loop will check convergence at the top of the next iteration.
  sleep "$POLL_INTERVAL"
done
