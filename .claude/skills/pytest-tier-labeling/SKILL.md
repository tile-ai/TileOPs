---
name: pytest-tier-labeling
description: Add or repair pytest smoke/full/nightly tier markers for tests under tests/. Use when users add new test files or new test cases in tests/, when new tests are missing tier labels, or when test collection fails due to explicit tier rules.
allowed-tools: Read, Grep, Glob, Bash, Edit
---

# Pytest Tier Labeling

> Apply this skill only to files under `tests/`. Do not use it for `benchmarks/`.

______________________________________________________________________

## 1. Scope

- Only modify `tests/`.
- Do not add `smoke` / `full` / `nightly` markers to `benchmarks/`.

______________________________________________________________________

## 2. Tier Rules

### 2.1 One tier per collected test case

Every collected test case must have exactly one tier marker:

- `pytest.mark.smoke`
- `pytest.mark.full`
- `pytest.mark.nightly`

### 2.2 Parameterized tests

For `FixtureBase.PARAMS` and `pytest.param(...)` cases:

- The first non-`xfail` case must be `smoke`
- The remaining normal coverage cases should be `full`
- Use `nightly` only for intentionally heavy or exhaustive coverage

### 2.3 Non-parameterized tests

For standalone test functions or class test methods:

- Add an explicit tier marker
- Default to `@pytest.mark.smoke`
- Use `full` or `nightly` only when the case is clearly heavier than smoke coverage

______________________________________________________________________

## 3. `tests/ops/` Hard Constraints

- Each test function must have exactly one `smoke` case
- That `smoke` case must be the first non-`xfail` case
- A `smoke` case must not be `xfail`

______________________________________________________________________

## 4. `tune=True` Constraints

If the parameterized test includes `tune`:

- `smoke` cases must use `tune=False`
- The first `tune=True` case must be marked `full`
- At most one `tune=True` case may be `full`

______________________________________________________________________

## 5. Editing Pattern

For parameter lists:

```python
pytest.param(..., marks=pytest.mark.smoke)
pytest.param(..., marks=pytest.mark.full)
pytest.param(..., marks=pytest.mark.nightly)
```

For standalone tests:

```python
@pytest.mark.smoke
def test_xxx(): ...
```

If you introduce `pytest.param(...)`, ensure the file imports `pytest`.

______________________________________________________________________

## 6. Workflow

1. Run:

```bash
python -m pytest tests --collect-only -q
```

2. Read the missing-tier or ordering errors.

1. Update only the affected files under `tests/`.

1. Re-run:

```bash
python -m pytest tests --collect-only -q
```

5. Stop only when collection succeeds.
