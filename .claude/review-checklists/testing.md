For PRs that add or modify files under `tests/`. Single focus: justify the test node growth declared in the PR body's `## Test node delta` section. Without this guard, parametrize stacks expand into performance sweeps and dilute every test run.

Load `.claude/domain-rules/testing-budget.md` and `docs/design/testing.md §Test case policy` before reviewing.

**Burden of proof flips above threshold.** If growth crosses the threshold below, the author justifies; reviewer silence is not approval.

#### Checklist

- [ ] **Trigger.** Compute `delta_pct = (HEAD − Base) / Base × 100` from the PR body's `## Test node delta`. If `delta_pct > 10%`, every check below is required; otherwise this file is informational.
- [ ] **Per-case purpose stated.** Each new case (or each new parametrize cell) serves exactly one of: dtype correctness / kernel-branch shape coverage / feature coverage / regression — per `docs/design/testing.md §Test case policy`. The PR body justification names which, with file:line.
- [ ] **No Cartesian-product expansion.** Reject parametrize stacks whose growth is the product of two or more axes' cardinalities without a per-cell rationale. Crossing axes is allowed only when each cell maps to a distinct code path the author can name; otherwise the stack is a performance sweep, not a UT.
- [ ] **Test layer hygiene.** Tests must not import from `benchmarks/` per `docs/design/trust-model.md`. Cross-layer touches are subject to strict-mode/principled-mode gating in `pre-review.md`.
