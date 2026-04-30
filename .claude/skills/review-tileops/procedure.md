1. **Read every changed source file in full.** The diff alone lacks surrounding context.
1. **Free-form review.** Before consulting any checklist, review the diff as code: logic errors, off-by-one and other edge cases, wrong/misused APIs, race conditions, resource leaks, broken invariants, error-handling gaps, perf regressions, dead code, confusing names, missing tests for the new behavior. Capture every concern as a candidate finding. The project-specific guards in step 3 are a regression net, not the boundary of your job.
1. **Apply each loaded project-specific guard.** Walk its bullets against the diff and the source files. Note any items that flag. Add to the candidate findings from step 2.
1. **Triage unresolved review threads.** For each thread, judge whether the developer's latest change resolves the concern. If unresolved, surface as an inline comment.
1. **Approval-gate decision.** If the diff touches `tests/` AND your draft event is `APPROVE`: read `.claude/review-checklists/approval-gate.md` and run every check. Downgrade to `REQUEST_CHANGES` if any fail.
1. **Compose inline comments** per `criteria.md` §2 — one per issue, format `<what is wrong and why> → <what to change>`. Name the function, variable, or pattern.
1. **Compose the summary** per `criteria.md` §3. Omit empty sections. A clean PR gets a single line: `Clean — no issues.`
1. **Submit one atomic review** per `criteria.md` §1, with the inline comments in the `comments=[…]` array. Follow `criteria.md` §4 hard rules.
