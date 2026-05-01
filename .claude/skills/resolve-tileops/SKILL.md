---
name: resolve-tileops
description: Per-round driver of stateful Claude-driven review-resolution on a tile-ai/TileOPs PR (developer side). Designed for /loop dynamic mode — re-fires until a terminal action. Assumes preflight.sh has run for this PR (see README.md). State persists in `.foundry/runs/{issue-<N> | pr-<PR>}/resolve/`.
---

## Notes

- GitHub auth: developer's own `gh` identity (not `gh-review`).
- Resolution work runs in **this Claude session** — no external subagent.

## Step 1: Pre-round

Run the pre-round script. It locates state, decides the action, and (on
`continue`) gathers the round's input snapshots.

```bash
PR=$ARGUMENTS
PRE=$(bash .claude/skills/resolve-tileops/round-pre.sh "$PR") || exit 1
ACTION=$(echo "$PRE" | jq -r .action)
ROUND=$(echo "$PRE" | jq -r .round)
RUN_DIR=$(echo "$PRE" | jq -r .run_dir)
SNAP=$(echo "$PRE" | jq -r .snap_prefix)
MESSAGE=$(echo "$PRE" | jq -r .message)
```

Branch on `ACTION`:

- `terminate-success`, `terminate-diverged`, `terminate-external` —
  write retrospective (see below), print `MESSAGE`, **return without
  ScheduleWakeup**.
- `idle` — print `MESSAGE`, ScheduleWakeup with `delaySeconds=180`,
  `prompt=/resolve-tileops <PR>`, `reason="PR #<PR> idle — polling"`. Return.
- `continue` — proceed to Step 2.

### Retrospective (terminal actions only)

Write a terse retrospective directly to `$RUN_DIR/retrospective.md` based
on this session's understanding of what happened across the loop's rounds.

Required:

- **Problem** (1–2 lines on core reviewer concerns)
- **Resolution** (`all-addressed` | `partial` | `unresolved`, one line)

Optional (only if substantive):

- **Approach** (1–2 lines on techniques applied)
- **Follow-up** (concrete deferred items, one per line)

Action-oriented; no long prose.

## Step 2: Load context

**Round 1 only** — read into your working context:

- `.claude/skills/resolve-tileops/procedure.md` — triage / fix / reply / resolve threads.
- `.claude/skills/resolve-tileops/criteria.md` — reply formats and hard rules.

Round 2+ rely on session memory.

**Every round** — read this round's inputs:

- `$RUN_DIR/inbox-history/round-NN.md` — this round's inbox guidance,
  if any (only present if `inbox.md` was non-empty when round-pre archived it).
- `$SNAP.new-reviews.json`
- `$SNAP.new-inline-comments.json`
- `$SNAP.unresolved-threads.json`
- `$SNAP.ci.json`

`round-pre.sh` only snapshots reviewer items NEW since the last processed
round, so round 2+ already sees only delta — no re-action on already-replied
threads needed.

## Step 3: Resolve

Apply `procedure.md`. Inbox guidance for this round (if any) overrides
default behavior where they conflict.

Do NOT post a structured trailer in any PR comment — the loop driver
detects round effects from GitHub state in Step 4.

## Step 4: Post-round

```bash
POST=$(bash .claude/skills/resolve-tileops/round-post.sh "$PR") || exit 1
PUSHED_SHA=$(echo "$POST" | jq -r .pushed_sha)
THREADS_RESOLVED=$(echo "$POST" | jq -r .threads_resolved)
OPEN_AFTER=$(echo "$POST" | jq -r .open_after)
```

Print: `Round $ROUND done — pushed=$PUSHED_SHA, threads_resolved=$THREADS_RESOLVED, open_after=$OPEN_AFTER.`

## Step 5: Self-schedule (only under /loop)

After a real work round, ScheduleWakeup:

- `prompt`: `/resolve-tileops <PR>`
- `delaySeconds`: `180`
- `reason`: `"PR #<PR> resolve round <ROUND> — pushed=<PUSHED_SHA>"`

Outside `/loop`: skip; just return.
