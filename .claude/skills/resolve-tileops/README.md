# resolve-tileops setup

This skill addresses reviewer feedback on a TileOPs PR. It runs in the
developer's own Claude session under `/loop` and drives a multi-round
resolve workflow.

## One-time setup

Nothing to do — the skill uses your normal `gh` auth. Verify:

```bash
command -v gh && gh auth status
```

The git clone must have a remote pointing to `tile-ai/TileOPs` (any name
works — `origin` in clones, `upstream` in forks, etc.).

## Driving a PR

Two steps. Preflight is the outer caller's responsibility — run it once
before starting the loop, then drive:

```bash
bash .claude/skills/resolve-tileops/preflight.sh <PR>   # idempotent; initializes state
/loop /resolve-tileops <PR>                              # then start the round loop
```

`preflight.sh` validates the env, parses `Closes #N` from the PR body to
choose a task root, and creates `.foundry/runs/{issue-<N>|pr-<PR>}/resolve/`
with the initial `meta.json` and an empty `inbox.md`. Re-running it is a
no-op once state exists.

Each round of `/loop` then runs `round-pre.sh` → Claude resolve work →
`round-post.sh`, scheduling the next wake-up via `ScheduleWakeup`.

## If preflight has not run

`round-pre.sh` errors out on the first invocation with:
`round-pre: no state for PR #<PR> — run preflight.sh first`.
Run preflight, then re-issue the `/loop` command.

## State layout

```
.foundry/runs/{issue-<N> | pr-<PR>}/   # task root, shared with reviewer loop
└── resolve/                            # this skill's state
    ├── meta.json                       # round counter, last processed IDs
    ├── inbox.md                        # user-editable side-channel; consumed each round
    ├── inbox-history/round-NN.md       # archived inbox per round
    ├── retrospective.md                # written on termination
    └── rounds/round-NN.{json,*.json}   # per-round snapshots + summary
```

The reviewer loop's state lives at the same task root under `review/`.

## User interaction during a loop

- Append to `inbox.md` → consumed next round, then archived under
  `inbox-history/round-NN.md`. Use this for one-shot guidance the skill
  should respect for the next round only (e.g. "the same blocker has been
  going back and forth — re-read `docs/design/architecture.md` before
  responding this time").
- Read `rounds/round-NN.json` for past round outcomes.
