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
