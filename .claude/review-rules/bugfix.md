# Review rule: bugfix

Applies to PRs prefixed `[Fix]` or `[BugFix]` — corrects incorrect behavior.

## Must check

- **Root cause stated**: PR description names the underlying defect, not just the symptom. Reviewer should be able to explain *why* the bug happened from the PR alone.
- **Regression test**: a test that fails on `main` and passes after the fix. Exception: pure infra/tooling fixes where authoring a test is disproportionate — call this out explicitly.
- **Minimal scope**: bugfix doesn't drag in refactors or feature work. A small adjacent cleanup is fine; reorganizing the surrounding module is not.
- **No silent test weakening**: don't gate the fix by relaxing an assertion.

## Sub-type: `[Fix][CI]` / `[Fix][Skills]` / `[Fix][Doc]`

Lower bar on regression test (CI/skills/doc bugs are often hard to test). Demand a clear repro description in the PR body instead.

## Don't gate on

- Refactoring opportunities the reviewer notices around the fix site — file as a follow-up issue.
- Whether the test name follows a particular convention beyond what `tests/` already does.

## Hard rejects

- Claims to fix a bug but adds no test and gives no testability rationale.
- Fix mutates `tileops/manifest/` to make code "match" — that inverts the trust model (`.claude/rules/manifest-trust-model.md`).
