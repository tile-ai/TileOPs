---
name: review-tileops-loop
description: Run one round of stateful Codex-driven review on a tile-ai/TileOPs PR. State persists under <task_root>/review/ via foundry helpers. Designed for /loop dynamic mode — re-fires until termination.
---

## Input

`$ARGUMENTS`: integer PR number in `tile-ai/TileOPs`. E.g. `1122`.

- Under `/loop` for auto-continue: `/loop /review-tileops-loop 1122`.
- Bare for a single round: `/review-tileops-loop 1122`.

## Constants

- `REPO=tile-ai/TileOPs`
- `MAX_ROUNDS=20`
- `REQUEST_CHANGES_DELAY_S=180` (3 min — fast polling for developer fix)
- Round reviewer: Codex (`codex --dangerously-bypass-approvals-and-sandbox exec ...`).
- Reviewer GitHub identity: see `.claude/skills/review-tileops/README.md`. Caller must run `bash .claude/skills/review-tileops/preflight.sh <PR>` once before the first round.

## Step 0: Wire foundry task root

```bash
PR=$ARGUMENTS                                        # must be integer
REPO_PATH=$(git rev-parse --show-toplevel)
FOUNDRY_ROOT="${FOUNDRY_ROOT:-/home/cy/TileOps-dev/foundry}"  # caller may override

# Resolve task root via foundry helper (issue-N when PR has Closes #N, else pr-M).
TASK_ROOT=$("$FOUNDRY_ROOT/scripts/task-root.sh" "$PR")
RUN_DIR="$TASK_ROOT/review"
META="$RUN_DIR/meta.json"
mkdir -p "$RUN_DIR/rounds" "$RUN_DIR/inbox-history"

# Reviewer auth: caller's preflight already ran; just re-export GH_CONFIG_DIR.
export GH_CONFIG_DIR="$TILEOPS_REVIEW_GH_CONFIG_DIR"

# Initialize task-level meta if missing (foundry pipeline didn't run for PR-only tasks).
"$FOUNDRY_ROOT/scripts/task-meta.sh" --task-root "$TASK_ROOT" init 2>/dev/null || true
```

## Step 0b: Init loop-local meta

```bash
if [ ! -f "$META" ]; then
  sed "s/__PR_NUMBER__/$PR/g" \
    .claude/skills/review-tileops-loop/policy-template.md > "$RUN_DIR/policy.md"
  : > "$RUN_DIR/inbox.md"
  jq -n --arg pr "$PR" --arg repo "$REPO" '{
    pr_number:($pr|tonumber), repo:$repo,
    codex_session_id:null, status:"active",
    round:0, max_rounds:20,
    last_reviewed_sha:null, last_reviewed_at:null,
    last_human_comment_id:0, last_codex_event:null,
    consecutive_codex_failures:0,
    last_criteria_mtime:0
  }' > "$META"
fi
```

Print: `Review loop for PR #<PR> — round <current+1>/20.   task_root=<TASK_ROOT>`.

______________________________________________________________________

## Step 1a: Termination check (before any work)

```bash
PR_JSON=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid,isDraft)
PR_STATE=$(echo "$PR_JSON" | jq -r .state)
HEAD_SHA=$(echo "$PR_JSON" | jq -r .headRefOid)

# Reviewer authors as Ibuki-wind-equivalent — gh now runs against the reviewer config dir.
# Read latest non-reviewer human comment id (issue + review threads).
REVIEWER_LOGIN=$(gh api user --jq .login)
LATEST_ISSUE_C=$(gh api "repos/$REPO/issues/$PR/comments" \
  --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\")|.id]|max // 0")
LATEST_REVIEW_C=$(gh api "repos/$REPO/pulls/$PR/comments" \
  --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\")|.id]|max // 0")
LATEST_HUMAN_ID=$(( LATEST_ISSUE_C > LATEST_REVIEW_C ? LATEST_ISSUE_C : LATEST_REVIEW_C ))

ROUND=$(jq -r .round "$META")
LAST_SHA=$(jq -r .last_reviewed_sha "$META")
LAST_EVENT=$(jq -r .last_codex_event "$META")
LAST_HUMAN_ID_PREV=$(jq -r .last_human_comment_id "$META")
```

First match wins; on match: set `meta.status`, run retrospective (Step 1b), print termination message, return WITHOUT `ScheduleWakeup`.

| #   | Condition                                                                                | status     | Message                                        |
| --- | ---------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------- |
| 1   | `PR_STATE` ∈ {`MERGED`,`CLOSED`}                                                         | `external` | "PR #$PR is $PR_STATE — stopping."             |
| 2   | `ROUND >= MAX_ROUNDS`                                                                    | `diverged` | "Reached max rounds — human attention needed." |
| 3   | `LAST_EVENT==APPROVE` AND `HEAD_SHA==LAST_SHA` AND `LATEST_HUMAN_ID==LAST_HUMAN_ID_PREV` | `success`  | "PR #\$PR converged."                          |

`PR_STATE==DRAFT` does not stop.

## Step 1b: Retrospective (on termination)

Skip if `codex_session_id` is null or status is `error`. Otherwise resume the same session for a terse retrospective.

```bash
SID=$(jq -r .codex_session_id "$META")
STATUS=$(jq -r .status "$META")
if [ "$SID" != "null" ] && [ -n "$SID" ] && [ "$STATUS" != "error" ]; then
  RETRO_PROMPT='Loop terminating. Output ONLY a terse retrospective in markdown — no preamble, no closing remarks.

Required:
1. **Problem** — 1–2 lines: core issue with this PR.
2. **Resolution** — `resolved` | `partial` | `unresolved`, one line on what is done.

Optional (omit if not substantive):
3. **Technical approach** — 1–2 lines on techniques applied.
4. **Follow-up** — concrete next-step items, one per line.

Action-oriented. No long prose.'

  codex --dangerously-bypass-approvals-and-sandbox exec resume "$SID" \
    --output-last-message "$RUN_DIR/retrospective.md" \
    --cd "$REPO_PATH" \
    "$RETRO_PROMPT" >/dev/null 2>&1 || true
fi

echo "=== Retrospective for PR #$PR (status=$STATUS) ==="
cat "$RUN_DIR/retrospective.md" 2>/dev/null || echo "(no retrospective written)"
echo "==="
```

______________________________________________________________________

## Step 2: Gather inputs

```bash
NEXT_ROUND=$((ROUND + 1))
N=$(printf '%02d' "$NEXT_ROUND")
SNAP="$RUN_DIR/rounds/round-$N"

# Diff: incremental if we have a previous sha, else full.
git -C "$REPO_PATH" fetch upstream "pull/$PR/head:review-pr-$PR-head" -f >/dev/null 2>&1
if [ "$LAST_SHA" != "null" ] && [ -n "$LAST_SHA" ]; then
  git -C "$REPO_PATH" diff "$LAST_SHA".."$HEAD_SHA" > "$SNAP.incremental.diff"
else
  gh pr diff "$PR" --repo "$REPO" > "$SNAP.full.diff"
fi

# Unresolved review threads.
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
  > "$SNAP.unresolved-threads.json"

# New non-reviewer human comments since last round.
gh api "repos/$REPO/issues/$PR/comments" \
  --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\" and .id>$LAST_HUMAN_ID_PREV)|{id,user:.user.login,body,created_at}]" \
  > "$SNAP.new-issue-comments.json"
gh api "repos/$REPO/pulls/$PR/comments" \
  --jq "[.[]|select(.user.type==\"User\" and .user.login!=\"$REVIEWER_LOGIN\" and .id>$LAST_HUMAN_ID_PREV)|{id,user:.user.login,path,line,body,created_at}]" \
  > "$SNAP.new-review-comments.json"

# CI.
gh pr checks "$PR" --repo "$REPO" --json name,state,conclusion \
  > "$SNAP.ci.json" 2>/dev/null || echo '[]' > "$SNAP.ci.json"
```

______________________________________________________________________

## Step 3: Compose prompt

```bash
INBOX_BLOCK="(none)"
if [ -s "$RUN_DIR/inbox.md" ]; then
  INBOX_BLOCK=$(cat "$RUN_DIR/inbox.md")
  mv "$RUN_DIR/inbox.md" "$RUN_DIR/inbox-history/round-$N.md"
  : > "$RUN_DIR/inbox.md"
fi

# Round 1: always inline criteria. Round 2+: inline only if criteria.md was edited
# since last inclusion (mtime check). Otherwise rely on Codex session memory.
CRITERIA_PATH=.claude/skills/review-tileops/criteria.md
CRITERIA_MTIME=$(stat -c %Y "$CRITERIA_PATH")
LAST_CRITERIA_MTIME=$(jq -r '.last_criteria_mtime // 0' "$META")
if [ "$NEXT_ROUND" -eq 1 ] || [ "$CRITERIA_MTIME" -gt "$LAST_CRITERIA_MTIME" ]; then
  INCLUDE_CRITERIA=1
  jq --argjson m "$CRITERIA_MTIME" '.last_criteria_mtime=$m' "$META" \
    > "$META.tmp" && mv "$META.tmp" "$META"
else
  INCLUDE_CRITERIA=0
fi
```

Write `$SNAP.prompt.md` with this template:

```
# Review Loop — PR #<PR>, Round <NEXT_ROUND>/<MAX_ROUNDS>

## Review criteria (shared spec)
[if INCLUDE_CRITERIA=1]  <verbatim contents of .claude/skills/review-tileops/criteria.md>
[else]                   Already loaded in an earlier round; refer to your conversation history. Unchanged since.

## Per-round policy
<verbatim contents of $RUN_DIR/policy.md>

## Temporary guidance for this round (extra-prompt override)
<INBOX_BLOCK>

## PR state
- HEAD sha: <HEAD_SHA>
- PR state: <PR_STATE>
- CI: <derived from $SNAP.ci.json>
- Previous reviewed sha: <LAST_SHA or "(first round)">
- Previous review event: <LAST_EVENT or "(first round)">

## Inputs (file paths — already prepared, do NOT re-fetch)
- Diff: <SNAP.incremental.diff or SNAP.full.diff>
- Unresolved threads: <SNAP.unresolved-threads.json>
- New human comments: <SNAP.new-issue-comments.json>, <SNAP.new-review-comments.json>

## Task
Read the criteria, policy, and temporary guidance above. Read the snapshot
files listed under Inputs (in particular the diff and any changed source
files referenced therein, in full). Apply criteria.md priorities with policy
and temporary guidance overriding where they conflict.

Submit ONE atomic review on tile-ai/TileOPs PR <PR> via gh per criteria §4.
The summary body MUST end with the trailer line specified by the policy.
Echo that trailer line back to stdout after submission.
```

______________________________________________________________________

## Step 4: Invoke Codex (retry up to 3)

```bash
SID=$(jq -r .codex_session_id "$META")
LASTMSG="$SNAP.codex-last-message.txt"
EVENTS="$SNAP.codex-events.jsonl"

run_codex() {
  if [ "$SID" = "null" ] || [ -z "$SID" ]; then
    codex --dangerously-bypass-approvals-and-sandbox exec \
      --json --output-last-message "$LASTMSG" --cd "$REPO_PATH" \
      "$(cat "$SNAP.prompt.md")" > "$EVENTS" 2>&1
  else
    codex --dangerously-bypass-approvals-and-sandbox exec resume "$SID" \
      --json --output-last-message "$LASTMSG" --cd "$REPO_PATH" \
      "$(cat "$SNAP.prompt.md")" > "$EVENTS" 2>&1
  fi
}

ATTEMPT=0
until run_codex && [ -s "$LASTMSG" ]; do
  ATTEMPT=$((ATTEMPT+1))
  if [ "$ATTEMPT" -ge 3 ]; then
    jq '.status="error"' "$META" > "$META.tmp" && mv "$META.tmp" "$META"
    echo "Codex failed 3 times — stopping." >&2; exit 1
  fi
  sleep 10
done

# Capture session UUID after first round.
if [ "$SID" = "null" ] || [ -z "$SID" ]; then
  SID=$(tail -n 50 ~/.codex/session_index.jsonl | jq -rs 'sort_by(.updated_at)|last|.id')
  jq --arg s "$SID" '.codex_session_id=$s' "$META" > "$META.tmp" && mv "$META.tmp" "$META"
fi
```

______________________________________________________________________

## Step 5: Validate output contract

```bash
REVIEW_BODY=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq "[.[]|select(.user.login==\"$REVIEWER_LOGIN\")]|sort_by(.submitted_at)|last|.body")
TRAILER=$(echo "$REVIEW_BODY" | grep -oE '<!-- review-loop: event=(APPROVE|REQUEST_CHANGES); blockers=[0-9]+; sha=[a-f0-9]{7} -->' | tail -1)

if [ -z "$TRAILER" ]; then
  ATTEMPT=$((ATTEMPT+1))
  if [ "$ATTEMPT" -ge 3 ]; then
    jq '.status="error"' "$META" > "$META.tmp" && mv "$META.tmp" "$META"
    echo "Codex output violated contract 3× — stopping." >&2; exit 1
  fi
  # Re-invoke Step 4 with a reminder appended to the prompt.
fi

EVENT=$(echo "$TRAILER" | sed -n 's/.*event=\([A-Z_]*\).*/\1/p')
BLOCKERS=$(echo "$TRAILER" | sed -n 's/.*blockers=\([0-9]*\).*/\1/p')
```

`EVENT` MUST be `APPROVE` or `REQUEST_CHANGES`.

______________________________________________________________________

## Step 6: Finalize round

```bash
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

jq -n --argjson r "$NEXT_ROUND" --arg now "$NOW" \
  --arg sha_b "${LAST_SHA:-null}" --arg sha_a "$HEAD_SHA" \
  --arg ev "$EVENT" --argjson bl "$BLOCKERS" \
  '{round:$r, finished_at:$now, head_sha_before:$sha_b, head_sha_after:$sha_a,
    codex_event:$ev, blockers_after:$bl}' > "$SNAP.json"

jq --argjson r "$NEXT_ROUND" --arg sha "$HEAD_SHA" --arg now "$NOW" \
   --argjson hid "$LATEST_HUMAN_ID" --arg ev "$EVENT" \
  '.round=$r | .last_reviewed_sha=$sha | .last_reviewed_at=$now
   | .last_human_comment_id=$hid | .last_codex_event=$ev
   | .consecutive_codex_failures=0' \
  "$META" > "$META.tmp" && mv "$META.tmp" "$META"

# Mirror to task-level meta: rounds counter + last event.
"$FOUNDRY_ROOT/scripts/task-meta.sh" --task-root "$TASK_ROOT" set-rounds review "$NEXT_ROUND" 2>/dev/null || true
"$FOUNDRY_ROOT/scripts/task-meta.sh" --task-root "$TASK_ROOT" set-event review "$EVENT" 2>/dev/null || true
```

Print: `Round $NEXT_ROUND done — event=$EVENT, blockers=$BLOCKERS, sha=${HEAD_SHA:0:7}.`

______________________________________________________________________

## Step 7: Self-schedule (only under /loop)

- `EVENT==APPROVE` → run **checklist introspection** (Step 7a), then exit. Do NOT call `ScheduleWakeup`. Print
  `Approved — loop ends. Checklist suggestions: $RUN_DIR/checklist-suggestions.md` and return.
- `EVENT==REQUEST_CHANGES` → call `ScheduleWakeup`:
  - `prompt`: `/loop /review-tileops-loop <PR>`
  - `delaySeconds`: `REQUEST_CHANGES_DELAY_S` (180)
  - `reason`: `"PR #<PR> round <NEXT_ROUND> — REQUEST_CHANGES; poll for fix"`

Outside `/loop`: skip; just return.

### Step 7a: Checklist introspection (APPROVE only)

Resume the Codex session and write `$RUN_DIR/checklist-suggestions.md`. Skip if `codex_session_id` is null or status is `error`. The skill never edits checklists; the file is for the user to apply by hand.

```bash
SID=$(jq -r .codex_session_id "$META")
STATUS=$(jq -r .status "$META")
if [ "$SID" != "null" ] && [ -n "$SID" ] && [ "$STATUS" != "error" ]; then
  INTROSPECT_PROMPT='Look back across all rounds of this PR (including fixed findings). Propose at most THREE bullets for `.claude/review-checklists/` that, in hindsight, would have caught a finding faster.

Output ONLY (no preamble, no closing):

## Proposed checklist additions

- **Target file**: `.claude/review-checklists/<file>.md`
  **Proposed bullet**: (one line, bold lead phrase, no [REQ]/[REC] tag)
  **Why this PR caught it**: (one line, round + finding)

Each proposal must be:
- **Executable** — pass/fail decidable without further interpretation.
- **Lean** — one sentence, no rationale, no examples.
- **Not implementation-coupled** — survives 5 future implementations (no file paths, function names, refactor-prone class names).

If nothing qualifies, output exactly: No proposed additions.'

  codex --dangerously-bypass-approvals-and-sandbox exec resume "$SID" \
    --output-last-message "$RUN_DIR/checklist-suggestions.md" \
    --cd "$REPO_PATH" \
    "$INTROSPECT_PROMPT" >/dev/null 2>&1 || true
fi
```

______________________________________________________________________

## Files written per run

```
<task_root>/review/
├── meta.json                       # loop-local: round, codex_session_id, last_*
├── policy.md                       # user-editable; applies next round
├── inbox.md                        # user-editable; consumed each round
├── inbox-history/round-NN.md       # archived inbox per round
├── retrospective.md                # written on termination
├── checklist-suggestions.md        # written on APPROVE (Step 7a)
└── rounds/
    ├── round-NN.json               # round summary
    ├── round-NN.prompt.md          # prompt sent to Codex
    ├── round-NN.incremental.diff   # or .full.diff on round 1
    ├── round-NN.unresolved-threads.json
    ├── round-NN.new-issue-comments.json
    ├── round-NN.new-review-comments.json
    ├── round-NN.ci.json
    ├── round-NN.codex-last-message.txt
    └── round-NN.codex-events.jsonl
```

`<task_root>/meta.json` (task-level, foundry-managed) gets `stages.review.rounds` and `stages.review.last_event` updated each round via `task-meta.sh`.

## User interaction

- Edit `<task_root>/review/policy.md` anytime → applies next round.
- Append to `<task_root>/review/inbox.md` → consumed next round, then archived.
- Read `<task_root>/review/rounds/round-NN.json` for past outcomes.
