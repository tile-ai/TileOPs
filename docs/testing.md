# Testing and Benchmarking

Tests and benchmarks are separated by concern: `pytest tests/` validates correctness only; `pytest benchmarks/` runs profiling only and auto-generates `profile_run.log`.

## Core Abstractions

| Class             | Location                  | Role                                                                                                                        |
| ----------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `FixtureBase`     | `tests/test_base.py`      | Metaclass-based decorator that applies `pytest.mark.parametrize` from a `PARAMS` class attribute.                           |
| `TestBase`        | `tests/test_base.py`      | ABC with `gen_inputs()`, `ref_program()`, `check()`, `check_fn()`. Each op subclasses this.                                 |
| `BenchmarkBase`   | `benchmarks/benchmark.py` | ABC wrapping a `TestBase` instance. Subclass implements `calculate_flops()` and `calculate_memory()`. Provides `profile()`. |
| `BenchmarkReport` | `benchmarks/benchmark.py` | Static collector — `record()` stores results, `dump()` writes markdown, `clear()` resets.                                   |

## Test/Benchmark Pattern

```python
# tests/ops/test_mha.py
class MhaFwdFixture(FixtureBase):
    PARAMS = [("batch, seq_len, heads, dim, causal, dtype, tune", [...])]


class MhaFwdTest(TestBase):
    def gen_inputs(self): ...
    def ref_program(self, q, k, v): ...


@MhaFwdFixture
def test_mha_fwd(batch, seq_len, heads, dim, causal, dtype, tune):
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    op = MultiHeadAttentionFwdOp(...)
    test.check(op, *test.gen_inputs())


# benchmarks/ops/bench_mha.py
class MhaFwdBenchmark(BenchmarkBase):
    def calculate_flops(self): ...
    def calculate_memory(self): ...


@MhaFwdFixture  # reuses the same parametrize decorator
def test_mha_fwd_bench(batch, seq_len, heads, dim, causal, dtype, tune):
    test = MhaFwdTest(batch, heads, seq_len, dim, causal, dtype)
    bm = MhaFwdBenchmark(test)
    inputs = test.gen_inputs()
    op = MultiHeadAttentionFwdOp(...)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mha_fwd", locals(), result, tag="tileops")
```

## Unit Test Requirements

**Framework:** pytest. **Location:** `tests/ops/`.

Each op defines a `TestBase` subclass with `gen_inputs()` and `ref_program()`.

### Tolerance

- Use `torch.testing.assert_close` for floating-point verification:
  - **FP16**: `rtol=1e-3`, `atol=1e-3`
  - **BF16**: `rtol=1.6e-2`, `atol=1.6e-2`
- Use exact comparison (`torch.equal`) for non-floating outputs (bool, masks, index tensors).

### Coverage Rules

- Tests must cover FP16 and BF16 data types.
- Tests must parameterize over common shapes (batch size, heads, sequence length).
- Tests must encode the dtype contract: supported dtypes are covered, unsupported dtypes are rejected, output dtypes are asserted when they differ from input.
- GPU-dependent tests must run on a real machine with host-visible CUDA devices. Sandbox-only results are not final correctness evidence.

### Infrastructure Rules

- Changes to shared test infrastructure (`tests/test_base.py`, common fixtures, shared comparators) must preserve existing default semantics unless all affected tests are migrated in the same PR.
- If a PR touches shared test infrastructure, run a broader `pytest -m smoke` pass before merge.
- Run full targeted test files for the affected op family on a real GPU before claiming readiness.

## Unit-Test Policy

This section defines principles that govern unit-test scope and growth for operator tests in `tests/ops/`. The goal is to keep the test suite effective without unbounded expansion.

### Allowed purposes

Unit tests exist to verify **correctness** of operator implementations. Each parameterized case must serve exactly one of these purposes:

1. **Dtype correctness** — verify the operator produces correct results for a supported dtype.
1. **Shape coverage** — verify the operator handles a representative shape (edge case, typical workload, or boundary condition).
1. **Feature coverage** — verify a feature flag or code path (e.g., `causal=True`, `tune=True`).
1. **Regression** — reproduce a specific bug that was fixed (must reference the issue or PR in the PARAMS comment).

Do not add parameterized cases for performance exploration, autotuning sweeps, or shape combinations that exercise the same code path as an existing case.

### Dtype risk classes

Not all dtypes need equal coverage. Group dtypes by risk class:

| Risk class | Dtypes                 | Required smoke cases | Rationale                                               |
| ---------- | ---------------------- | -------------------- | ------------------------------------------------------- |
| Primary    | float16, bfloat16      | At least 1 each      | Core production dtypes; different tolerance budgets     |
| Secondary  | float32, float8_e4m3fn | 1 total              | Used for reference or emerging; share accumulation path |
| Edge       | int8, bool, int32      | Only if op supports  | Discrete types; one case per supported type suffices    |

When an operator supports both primary dtypes, the smoke tier must include at least one case for each. Secondary and edge dtypes may be full-tier only.

### Shape coverage

Shape parameters should cover three categories with minimal redundancy:

1. **Minimal** — smallest valid shape (e.g., batch=1, seq_len=1, heads=1). Catches off-by-one and boundary bugs.
1. **Typical** — a representative real-workload shape (e.g., LLaMA-family dimensions). Catches mainstream correctness issues.
1. **Stress** — a large or unusual shape (non-power-of-two, prime dimensions, large batch). Catches tile-boundary and overflow bugs.

Adding a fourth or fifth shape in the same category requires justification in the PARAMS comment or PR description. The question to answer: "What distinct code path or boundary does this shape exercise that the existing shapes do not?"

### Manifest separation

Test parameter lists (PARAMS) define **what to test**. Op manifests (`ops_manifest.yaml`) define **what the op supports**. These are separate concerns:

- PARAMS is a curated subset chosen for correctness coverage, not an exhaustive enumeration of manifest workloads.
- Do not auto-generate PARAMS from manifest entries. Manifest workloads are designed for benchmarking and documentation, not for testing economy.
- If a manifest workload reveals a bug, add a targeted regression case to PARAMS, not the entire workload list.

### Growth justification

When adding new parameterized cases to an existing test:

- Each new case must state its **purpose** (dtype, shape, feature, or regression) either in a PARAMS comment or the PR description.
- If the total parameterized case count for a single test function grows beyond 20, the PR must explain why the additional coverage is necessary.
- Use the `scripts/test_node_delta.py` script (see below) to measure test node growth and include the delta in the PR description when growth exceeds 10%.
- Prefer adding a new targeted test function over inflating an existing function's PARAMS when testing a genuinely different behavior.

### Test node growth detection

The `scripts/test_node_delta.py` script compares pytest test-node counts between the current branch and main. Run it before submitting a PR that adds or modifies tests.

**Auto-detect changed test files:**

```bash
python scripts/test_node_delta.py
```

**Check specific files:**

```bash
python scripts/test_node_delta.py tests/ops/test_foo.py tests/ops/test_bar.py
```

**Compare against a different base branch:**

```bash
python scripts/test_node_delta.py --base origin/release
```

The script outputs a table showing per-file base count, HEAD count, and delta. It always exits 0 (non-blocking). New files that do not exist on the base branch are labeled `(new)`.

Include the output in your PR description when the total growth exceeds 10%.

## Benchmark Requirements

**Framework:** `benchmarks.benchmark.BenchmarkBase`. **Location:** `benchmarks/ops/`.

**Execution:** `pytest benchmarks/` auto-generates `profile_run.log` (markdown format).

### Metrics (all required)

- Latency (ms)
- TFLOPS (Tera Floating-point Operations Per Second)
- DRAM Bandwidth (GB/s)

### Reporting Rules

- Numbers must come from a real GPU machine, not a sandbox.
- Include small, medium, and large representative shapes.
- Do not cherry-pick favorable shapes; report regressions as-is.
- Run the targeted correctness suite on the same GPU before reporting benchmark numbers.
- `BenchmarkReport.record()` first argument may be the Op instance or a string name; stay consistent within a given benchmark file.
- `calculate_flops()` and `calculate_memory()` should return numeric values when the metric is available; return `None` only if the metric is not applicable, in which case it will be omitted from the report.
- Every benchmark must record at least one non-`"tileops"` baseline. Use existing tags (`"baseline"`, `"torch"`, `"FA3"`, `"fla"`, `"triton"`) and avoid introducing ad-hoc tags without updating downstream consumers.
