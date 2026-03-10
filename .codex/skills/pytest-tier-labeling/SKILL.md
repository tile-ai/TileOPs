---
name: pytest-tier-labeling
description: Add or repair pytest smoke/full/nightly tier markers for tests under tests/. Use when users add new test files or new test cases in tests/, ask to tag new tests, fix missing pytest tier labels, or enforce the repository's test-tier collection rules. Do not use for benchmarks/.
---

# When to use

- A user asks to add `pytest.mark.smoke`, `pytest.mark.full`, or `pytest.mark.nightly`.
- `python -m pytest tests --collect-only -q` fails because tests are missing tier markers.
- You are editing tests under `tests/` and need to align them with the repository's explicit tier rules.

# Scope

- Only modify files under `tests/`.
- Do not add tier markers to `benchmarks/`.

# Rules

## One marker per test case

- Every collected test case must have exactly one of:
  - `pytest.mark.smoke`
  - `pytest.mark.full`
  - `pytest.mark.nightly`

## Parameterized tests

- For `FixtureBase.PARAMS` or `pytest.param(...)` cases:
  - Mark the first non-`xfail` case as `smoke`.
  - Mark the remaining normal coverage cases as `full`.
  - Use `nightly` only for intentionally heavy or exhaustive cases.

## Non-parameterized tests

- For standalone test functions or class test methods in `tests/ops/`:
  - Add an explicit tier marker.
  - Default to `@pytest.mark.smoke` unless there is a clear reason it belongs in `full` or `nightly`.

## `tests/ops/` hard constraints

- Each test function must have exactly one `smoke` case.
- That `smoke` case must be the first non-`xfail` case.
- A `smoke` case must not also be `xfail`.

## `tune=True` constraint

- If a parameterized test has a `tune` argument:
  - `smoke` cases must use `tune=False`.
  - The first `tune=True` case must be marked `full`.
  - At most one `tune=True` case may be `full`.

# Editing pattern

- Prefer `pytest.param(..., marks=pytest.mark.smoke)` / `full` / `nightly` inside parameter lists.
- For function-level markers, use decorators:

```python
@pytest.mark.smoke
def test_xxx(): ...
```

- If you introduce `pytest.param(...)` into a file that did not previously need it, ensure `import pytest` exists.

# Workflow

1. Run `python -m pytest tests --collect-only -q`.
1. Read the reported missing-marker errors.
1. Update only the affected files under `tests/`.
1. Re-run `python -m pytest tests --collect-only -q`.
1. Stop only when collection succeeds.

# Output expectations

- Report which `tests/` files were changed.
- Report the result of `python -m pytest tests --collect-only -q`.
