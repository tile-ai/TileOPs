---
name: resolve-tileops
description: Run one round of stateful Claude-driven review-resolution on a tile-ai/TileOPs PR (developer side). Designed for /loop dynamic mode — re-fires until termination conditions are met. State persists in the repo's .foundry/runs/ (gitignored).
---

## Input

`$ARGUMENTS`: integer PR number in `tile-ai/TileOPs`. E.g. `1072`.

Invoke under `/loop` for auto-continue: `/loop /resolve-tileops 1072`.
Invoke bare for a single round: `/resolve-tileops 1072`.

## Constants

- `REPO=tile-ai/TileOPs`
- `RUN_DIR=$REPO_PATH/.foundry/runs/resolve-pr-<N>`
- `MAX_ROUNDS=20`
- `POLL_INTERVAL_S=180` (3 min — same cadence as review-tileops-loop)
- GitHub: bare `gh` (developer's own identity, NOT `gh-review`).
- Resolution work runs in **this Claude session** — no external subagent.

## Preflight

```bash
command -v gh >/dev/null 2>&1 || { echo "missing gh" >&2; exit 1; }

REPO_PATH=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not in a git repo" >&2; exit 1; }

UPSTREAM_REMOTE=$(git -C "$REPO_PATH" remote -v | \
  awk '/tile-ai\/TileOPs(\.git)?[[:space:]]+\(fetch\)/ {print $1; exit}')
if [ -z "$UPSTREAM_REMOTE" ]; then
  echo "no git remote in $REPO_PATH points to tile-ai/TileOPs" >&2
  exit 1
fi
```

Each worktree owns its own `.foundry/runs/resolve-pr-<N>/`; `cd` into the
worktree before invoking.

______________________________________________________________________

## Step 0: Init

```bash
PR=$ARGUMENTS  # must be integer
RUN_DIR="$REPO_PATH/.foundry/runs/resolve-pr-$PR"
META="$RUN_DIR/meta.json"

if [ ! -f "$META" ]; then
  mkdir -p "$RUN_DIR/rounds" "$RUN_DIR/inbox-history"
  sed "s/__PR_NUMBER__/$PR/g" \
    "$REPO_PATH/.claude/skills/resolve-tileops/policy-template.md" > "$RUN_DIR/policy.md"
  : > "$RUN_DIR/inbox.md"
  jq -n --arg pr "$PR" --arg repo "$REPO" '{
    pr_number:($pr|tonumber), repo:$repo,
    status:"active",
    round:0, max_rounds:20,
    last_processed_review_id:0,
    last_processed_review_comment_id:0,
    last_pushed_sha:null
  }' > "$META"
fi
```

Print: `Resolve loop for PR #<N> — round <current+1>/20.`

______________________________________________________________________

## Step 1a: Termination check (before any work)

```bash
PR_JSON=$(gh pr view "$PR" --repo "$REPO" --json state,headRefOid,isDraft)
PR_STATE=$(echo "$PR_JSON" | jq -r .state)
HEAD_SHA=$(echo "$PR_JSON" | jq -r .headRefOid)

# Latest Ibuki-wind review state
LATEST_REVIEWER_STATE=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")] | sort_by(.submitted_at) | last | .state // "NONE"')

# Unresolved threads (count and full list)
UNRESOLVED=$(gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){ nodes{ isResolved } }
      }
    }
  }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')

# Latest reviewer-side IDs (for change detection)
LATEST_REVIEW_ID=$(gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')
LATEST_REVIEW_COMMENT_ID=$(gh api "repos/$REPO/pulls/$PR/comments" \
  --jq '[.[]|select(.user.login=="Ibuki-wind")|.id]|max // 0')

ROUND=$(jq -r .round "$META")
LAST_REVIEW_ID_PREV=$(jq -r .last_processed_review_id "$META")
LAST_REVIEW_COMMENT_ID_PREV=$(jq -r .last_processed_review_comment_id "$META")
LAST_PUSHED_SHA=$(jq -r .last_pushed_sha "$META")
```

First match wins; on match: set `meta.status`, run retrospective (Step 1b),
print termination message, return WITHOUT `ScheduleWakeup`.

| #   | Condition                                                                                                                                                               | status     | Message                                                         |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------- |
| 1   | `PR_STATE` ∈ {`MERGED`,`CLOSED`}                                                                                                                                        | `external` | "PR #$PR is $PR_STATE — stopping."                              |
| 2   | `ROUND >= MAX_ROUNDS`                                                                                                                                                   | `diverged` | "Reached max rounds — human attention needed."                  |
| 3   | `UNRESOLVED == 0` AND `LATEST_REVIEWER_STATE == "APPROVED"` AND `LATEST_REVIEW_ID == LAST_REVIEW_ID_PREV` AND `LATEST_REVIEW_COMMENT_ID == LAST_REVIEW_COMMENT_ID_PREV` | `success`  | "PR #\$PR converged — all threads resolved, reviewer approved." |

`PR_STATE==DRAFT` does not stop.

## Step 1b: Retrospective (on termination)

Skip if status is `error`. Otherwise, **Claude writes a terse retrospective
directly** to `$RUN_DIR/retrospective.md` based on this session's
understanding of what happened across the loop's rounds.

Required sections: **Problem** (1–2 lines on core reviewer concerns) and
**Resolution** (`all-addressed` | `partial` | `unresolved`, one line on
what's done). Optional (include only if substantive): **Approach** (1–2
lines on techniques applied) and **Follow-up** (concrete deferred items,
one per line). Action-oriented; no long prose.

```bash
echo "=== Retrospective for PR #$PR (status=$STATUS) ==="
cat "$RUN_DIR/retrospective.md" 2>/dev/null || echo "(no retrospective written)"
echo "==="
```

______________________________________________________________________

## Step 1c: Idle gate (skip work if no new feedback)

If `LATEST_REVIEW_ID == LAST_REVIEW_ID_PREV` AND
`LATEST_REVIEW_COMMENT_ID == LAST_REVIEW_COMMENT_ID_PREV`:

- No new reviewer activity since the last processed round. Do NOT
  increment round. Do NOT gather inputs. Do NOT do resolution work.
- Print: `No new review feedback — sleeping ${POLL_INTERVAL_S}s.`
- Under `/loop`: call `ScheduleWakeup` with `delaySeconds=POLL_INTERVAL_S`,
  `prompt=/resolve-tileops <PR>`,
  `reason="PR #<PR> idle — polling for new review"`. Return.
- Outside `/loop`: just return.

This gate is the polling mechanism. Each wake-up costs only a few `gh api`
calls + jq parsing — no Claude reasoning load, no resolution work.

______________________________________________________________________

## Step 2: Gather inputs

```bash
NEXT_ROUND=$((ROUND + 1))
N=$(printf '%02d' "$NEXT_ROUND")
SNAP="$RUN_DIR/rounds/round-$N"

# New reviewer summaries since last_processed_review_id
gh api "repos/$REPO/pulls/$PR/reviews" \
  --jq "[.[]|select(.user.login==\"Ibuki-wind\" and .id>$LAST_REVIEW_ID_PREV)|{id,state,body,submitted_at}]" \
  > "$SNAP.new-reviews.json"

# New inline comments since last_processed_review_comment_id
gh api "repos/$REPO/pulls/$PR/comments" \
  --jq "[.[]|select(.user.login==\"Ibuki-wind\" and .id>$LAST_REVIEW_COMMENT_ID_PREV)|{id,path,line,body,in_reply_to_id,created_at}]" \
  > "$SNAP.new-inline-comments.json"

# All currently unresolved threads (full snapshot — state, not delta)
gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){
          nodes{
            id isResolved
            comments(first:20){ nodes{ databaseId author{login} body path line } }
          }
        }
      }
    }
  }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
  --jq '.data.repository.pullRequest.reviewThreads.nodes|map(select(.isResolved==false))' \
  > "$SNAP.unresolved-threads.json"

# CI status
gh pr checks "$PR" --repo "$REPO" --json name,state,conclusion \
  > "$SNAP.ci.json" 2>/dev/null || echo '[]' > "$SNAP.ci.json"
```

______________________________________________________________________

## Step 3: Load procedure context

Read into your working context (you'll act on these in Step 4):

- `.claude/skills/resolve-tileops/procedure.md` — the resolution procedure
  (triage → fix → reply → resolve threads). Read in full on the **first
  round only**; subsequent rounds rely on this conversation's history.
- `.claude/skills/resolve-tileops/criteria.md` — reply formats and hard rules.
  Read on round 1.
- `$RUN_DIR/policy.md` — per-round policy, user-editable. Read every round.
- `$RUN_DIR/inbox.md` — one-shot extra guidance, consumed this round.

After reading inbox, archive it:

```bash
if [ -s "$RUN_DIR/inbox.md" ]; then
  mv "$RUN_DIR/inbox.md" "$RUN_DIR/inbox-history/round-$N.md"
  : > "$RUN_DIR/inbox.md"
fi
```

Also load the snapshots from Step 2:
`$SNAP.new-reviews.json`, `$SNAP.new-inline-comments.json`,
`$SNAP.unresolved-threads.json`, `$SNAP.ci.json`.

______________________________________________________________________

## Step 4: Execute resolution (Claude, in this session)

Apply the `resolve-tileops` procedure to the feedback gathered in Step 2:

1. **Triage** each new review summary item, new inline comment, and
   currently-unresolved thread → Accept / Reject / Defer. Bias toward
   Accept. (procedure §Triage)
1. **Apply fixes** for Accepted items: edit code, run pre-commit / linters
   if configured, commit with `[Chore][<scope>] address review feedback`,
   push to the PR branch. (procedure §Apply fixes)
1. **Reply** on every thread + summary item; **resolve** threads
   regardless of verdict. (procedure §Reply and resolve, criteria §1)

Per-round policy (`$RUN_DIR/policy.md`) and inbox guidance (consumed in
Step 3) override default behavior where they conflict.

Do NOT post a structured trailer in any PR comment — the loop driver
detects round effects from GitHub state in Step 5.

______________________________________________________________________

## Step 5: Detect round effects

```bash
NEW_HEAD_SHA=$(gh pr view "$PR" --repo "$REPO" --json headRefOid --jq .headRefOid)

if [ "$NEW_HEAD_SHA" != "$HEAD_SHA" ]; then
  PUSHED_SHA="$NEW_HEAD_SHA"
else
  PUSHED_SHA="none"
fi

NEW_UNRESOLVED=$(gh api graphql -f query='
  query($owner:String!,$repo:String!,$pr:Int!){
    repository(owner:$owner,name:$repo){
      pullRequest(number:$pr){
        reviewThreads(first:100){ nodes{ isResolved } }
      }
    }
  }' -F owner=tile-ai -F repo=TileOPs -F pr="$PR" \
  --jq '[.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)]|length')

THREADS_RESOLVED=$(( UNRESOLVED - NEW_UNRESOLVED ))
```

______________________________________________________________________

## Step 6: Finalize round

```bash
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

jq -n --argjson r "$NEXT_ROUND" --arg now "$NOW" \
  --arg sha_b "$HEAD_SHA" --arg sha_a "$NEW_HEAD_SHA" \
  --arg pushed "$PUSHED_SHA" \
  --argjson resolved "$THREADS_RESOLVED" \
  --argjson unresolved_after "$NEW_UNRESOLVED" \
  --arg reviewer_state "$LATEST_REVIEWER_STATE" \
  '{round:$r, finished_at:$now,
    head_sha_before:$sha_b, head_sha_after:$sha_a,
    pushed_sha:$pushed, threads_resolved:$resolved,
    unresolved_after:$unresolved_after,
    reviewer_state_before:$reviewer_state}' > "$SNAP.json"

# pushed_sha goes to meta only if a push actually happened
PUSHED_FOR_META="$LAST_PUSHED_SHA"
[ "$PUSHED_SHA" != "none" ] && PUSHED_FOR_META="$PUSHED_SHA"

jq --argjson r "$NEXT_ROUND" \
   --argjson rid "$LATEST_REVIEW_ID" \
   --argjson cid "$LATEST_REVIEW_COMMENT_ID" \
   --arg pushed "$PUSHED_FOR_META" \
  '.round=$r | .last_processed_review_id=$rid
   | .last_processed_review_comment_id=$cid
   | .last_pushed_sha=$pushed' \
  "$META" > "$META.tmp" && mv "$META.tmp" "$META"
```

Print: `Round $NEXT_ROUND done — pushed=${PUSHED_SHA}, threads_resolved=$THREADS_RESOLVED, open_after=$NEW_UNRESOLVED.`

______________________________________________________________________

## Step 7: Self-schedule (only under /loop)

After a real work round, schedule the next wake-up at `POLL_INTERVAL_S`
(180s) — same cadence as the idle gate and the reviewer loop's
`REQUEST_CHANGES_DELAY_S`. Aligning dev and reviewer cadences keeps the
ping-pong tight.

Call `ScheduleWakeup`:

- `prompt`: `/resolve-tileops <PR>`
- `delaySeconds`: `POLL_INTERVAL_S`
- `reason`: `"PR #<PR> resolve round <NEXT_ROUND> — pushed=<PUSHED_SHA>"`

Outside `/loop`: skip; just return.

______________________________________________________________________

## Files written per run

```
.foundry/runs/resolve-pr-<N>/
├── meta.json
├── policy.md                       # user-editable; applies next round
├── inbox.md                        # user-editable; consumed each round
├── inbox-history/round-NN.md       # archived inbox per round
├── retrospective.md                # written on termination
└── rounds/
    ├── round-NN.json               # summary
    ├── round-NN.new-reviews.json
    ├── round-NN.new-inline-comments.json
    ├── round-NN.unresolved-threads.json
    └── round-NN.ci.json
```

## User interaction

- Edit `policy.md` anytime → applies next round.
- Append to `inbox.md` → consumed next round, then archived.
- Read `rounds/round-NN.json` for past outcomes.
