# Review procedure

Single source of truth for "one review pass on one tile-ai/TileOPs PR". Followed verbatim by:

- `SKILL.md` — manual single-shot via Claude / Codex session.
- `loop.sh` — per-PR autonomous loop, each round.

Both modes wrap this procedure with mode-specific concerns (preflight, input gathering, round bookkeeping, trailer for the loop driver). The review behavior itself lives only here.

## Assumed inputs

The caller has already ensured:

- `preflight.sh` has run and `GH_CONFIG_DIR` is exported to the reviewer identity.
- The PR has been classified per `loading.yaml`; the list of applicable checklist files is known.
- The diff (full or incremental), unresolved review threads, recent non-reviewer comments, and CI status are accessible (either pre-fetched on disk or fetchable on demand).

## Steps

1. **Read every changed source file in full.** The diff alone lacks surrounding context.
1. **Apply each loaded checklist.** Walk its bullets against the diff and the source files. Note any items that flag.
1. **Triage unresolved review threads.** For each thread, judge whether the developer's latest change resolves the concern. If unresolved, surface as an inline comment.
1. **Approval-gate decision.** If the diff touches `tests/` AND your draft event is `APPROVE`: read `.claude/review-checklists/approval-gate.md` and run every check. Downgrade to `REQUEST_CHANGES` if any fail.
1. **Compose inline comments** per `criteria.md` §2 — one per issue, format `<what is wrong and why> → <what to change>`. Name the function, variable, or pattern.
1. **Compose the summary** per `criteria.md` §3. Omit empty sections. A clean PR gets a single line: `Clean — no issues.`
1. **Submit one atomic review** per `criteria.md` §1, with the inline comments in the `comments=[…]` array. Follow `criteria.md` §4 hard rules.
