**Free-form review (step 2) is the primary review step.** Project-specific guards (step 4) are a regression net applied last — they exist because the free-form pass is fallible at known classes of issue, NOT because they replace it. Skipping or shortening step 2 to lean on guards is a failure mode.

1. **Read every changed source file in full.** The diff alone lacks surrounding context.
1. **Free-form review (primary).** Flag: logic errors, edge cases, API misuse, races, resource leaks, broken invariants, error-handling gaps, perf regressions, dead code, unclear names, missing tests. This is where the bulk of your reasoning belongs.
1. **Triage unresolved review threads.** For each thread, judge whether the developer's latest change resolves the concern. If unresolved, surface as an inline comment.
1. **Apply each loaded project-specific guard (last, supplement).** Walk its bullets against the diff. Add anything new to the step-2 findings.
1. **Approval-gate decision.** If the diff touches `tests/` AND your draft event is `APPROVE`: read `.claude/review-checklists/approval-gate.md` and run every check. Downgrade to `REQUEST_CHANGES` if any fail.
1. **Compose inline comments** per `criteria.md` §2 — one per issue, format `<what is wrong and why> → <what to change>`. Name the function, variable, or pattern.
1. **Compose the summary** per `criteria.md` §3. Omit empty sections. A clean PR gets a single line: `Clean — no issues.`
1. **Submit one atomic review** per `criteria.md` §1, with the inline comments in the `comments=[…]` array. Follow `criteria.md` §4 hard rules.
