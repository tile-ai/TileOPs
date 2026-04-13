# Testing and Benchmarking

Tests and benchmarks are separated by concern: `pytest tests/` validates correctness only; `pytest benchmarks/` runs profiling only and auto-generates `profile_run.log`.

## Core Abstractions

| Class              | Location                                                          | Role                                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WorkloadBase`     | [`workloads/workload_base.py`](../workloads/workload_base.py)     | ABC defining `gen_inputs()`. Shared base for input generation used by both tests and benchmarks.                                                                                    |
| `FixtureBase`      | [`workloads/workload_base.py`](../workloads/workload_base.py)     | Metaclass-based decorator that applies `pytest.mark.parametrize` from a `PARAMS` class attribute or `get_params()` classmethod.                                                     |
| `TestBase`         | [`tests/test_base.py`](../tests/test_base.py)                     | Inherits `WorkloadBase`. Adds `ref_program()` and `check()`. Each op subclasses this for correctness testing.                                                                       |
| `BenchmarkBase[W]` | [`benchmarks/benchmark_base.py`](../benchmarks/benchmark_base.py) | Generic ABC parameterized by workload type `W` (a capability protocol, not `WorkloadBase`). Subclass implements `calculate_flops()` and `calculate_memory()`. Provides `profile()`. |
| `BenchmarkReport`  | [`benchmarks/benchmark_base.py`](../benchmarks/benchmark_base.py) | Static collector -- `record()` stores results, `dump()` writes markdown, `clear()` resets.                                                                                          |

## Wiring

Workload is defined once; test and benchmark each reference it but do not depend on each other:

- **Workload** (`workloads/`) — `WorkloadBase` subclass, defines `gen_inputs()`
- **Test** (`tests/ops/`) — inherits `(Workload, TestBase)`, adds `ref_program()` locally
- **Benchmark** (`benchmarks/ops/`) — composes workload via `BenchmarkBase(workload)`

Rules:

- **Fixture usage**: both tests and benchmarks can use `FixtureBase`, but params are usually defined per layer unless intentionally factored into a shared module
- **Dependency direction**: benchmark imports workload, never test
- **ref_program locality**: correctness oracle is defined in the test file, not in workload

## Tests

→ Trust boundary: [trust-model.md §Test](trust-model.md#test) | Rules: [testing-budget.md](../.claude/domain-rules/testing-budget.md)

**Framework:** pytest. **Location:** [`tests/ops/`](../tests/ops/).

### File checklist

1. **Workload class** in `workloads/` — subclass `WorkloadBase`, implement `gen_inputs()`.
1. **Fixture class** — subclass `FixtureBase`, define `PARAMS` with `smoke`/`full` marks.
1. **Test class** in `tests/ops/test_<op>.py` — inherit `(MyWorkload, TestBase)`, implement `ref_program()` locally.
1. **Test function** — `@YourFixture` decorated, call `test.check(op, *test.gen_inputs())`.

### Tolerance

- Use `torch.testing.assert_close` for floating-point verification:
  - **FP16**: `rtol=1e-3`, `atol=1e-3`
  - **BF16**: `rtol=1.6e-2`, `atol=1.6e-2`
- Use exact comparison (`torch.equal`) for non-floating outputs (bool, masks, index tensors).

### Coverage rules

- Tests must cover FP16 and BF16 data types.
- Tests must parameterize over common shapes (batch size, heads, sequence length).
- Tests must encode the dtype contract: supported dtypes are covered, unsupported dtypes are rejected, output dtypes are asserted when they differ from input.
- GPU-dependent tests must run on a real machine with host-visible CUDA devices. Sandbox-only results are not final correctness evidence.

### Test case policy

Each parameterized case must serve one of:

1. **Dtype correctness** — verify a supported dtype.
1. **Shape coverage** — verify a distinct code path (boundary, tile edge, alignment).
1. **Feature coverage** — verify a feature flag or mode (`causal=True`, `tune=True`).
1. **Regression** — reproduce a fixed bug (reference issue/PR in comment).

No performance exploration, autotune sweeps, or duplicate code-path coverage.

**Dtype coverage:** All supported dtypes must be tested. Smoke: cover each dtype with one typical shape. Full: cross-combinations only when the implementer can name the code path each guards.

**Shape coverage:** UT shapes target kernel implementation branches, not workload representativeness. Common kernel branch conditions:

- **Tile boundary** — shape not divisible by tile size (tail handling)
- **Vectorization alignment** — shape not aligned to vector width (scalar fallback)
- **Degenerate dimension** — size=1 (broadcast, squeeze paths)
- **Dispatch branch** — different shape ranges triggering different kernel variants

The implementer selects the smallest shape that triggers each branch. Do not generate test fixtures from [`ops_manifest.yaml`](../tileops/ops_manifest.yaml) workloads.

**Growth rules:**

- Each new case must state its purpose (dtype / shape / feature / regression) in a comment or PR description.
- Over 20 cases per test function: justify which code paths require the count.
- Prefer a new test function over inflating an existing one when testing genuinely different behavior.

### Test node growth detection

[`scripts/test_node_delta.py`](../scripts/test_node_delta.py) compares **pytest collected node count** (test cases after parametrize expansion) between current branch and main. Always exits 0 (non-blocking).

```bash
python scripts/test_node_delta.py                    # auto-detect changed test files
python scripts/test_node_delta.py tests/ops/test_foo.py  # specific files
python scripts/test_node_delta.py --base origin/release   # different base branch
```

- **No growth on existing files**: nothing to report.
- **Growth on existing files**: include script output and a one-line justification in PR description.
- **New test files only**: no delta to report — follow the policy above.

### Testing layers

| Layer             | Responsibility                                      | Shape source                                                  |
| ----------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| UT smoke/full     | Guard PR correctness                                | Implementer selects based on kernel code paths                |
| Nightly benchmark | Performance regression + typical/stress correctness | [`ops_manifest.yaml`](../tileops/ops_manifest.yaml) workloads |
| Local dev         | Performance tuning verification                     | Developer decides ad-hoc                                      |

### Infrastructure rules

- Changes to shared test infrastructure ([`tests/test_base.py`](../tests/test_base.py), common fixtures, shared comparators) must preserve existing default semantics unless all affected tests are migrated in the same PR.
- If a PR touches shared test infrastructure, run a broader `pytest -m smoke` pass before merge.
- Run full targeted test files for the affected op family on a real GPU before claiming readiness.

## Benchmarks

→ Trust boundary: [trust-model.md §Benchmark](trust-model.md#benchmark) | Rules: [benchmark.md](../.claude/domain-rules/benchmark.md)

**Framework:** `benchmarks.benchmark.BenchmarkBase`. **Location:** [`benchmarks/ops/`](../benchmarks/ops/).

**Execution:** `pytest benchmarks/` auto-generates `profile_run.log` (markdown format).

### Workload protocols

`BenchmarkBase[W]` is generic over workload type — different benchmarks depend on different workload capabilities, so the type parameter `W` is a capability protocol, not `WorkloadBase`. `WorkloadBase` remains the default in-repo implementation; the public benchmark API is defined by these protocols in `benchmarks/benchmark_base.py`:

| Protocol                  | Requires          | Use when                                            |
| ------------------------- | ----------------- | --------------------------------------------------- |
| `ShapeDtypeWorkload`      | `shape`, `dtype`  | Helper only reads metadata (e.g. `roofline_vars()`) |
| `InputGeneratingWorkload` | `gen_inputs()`    | Code only needs input generation                    |
| `BenchmarkWorkload`       | Both of the above | Code needs metadata and input generation            |

For benchmark-specific metadata (e.g. `m/n/k` for GEMM), define a dedicated protocol for that benchmark family.

### File checklist

1. **Workload** — any object satisfying the required protocol (e.g. `ShapeDtypeWorkload`). Often a `WorkloadBase` subclass from `workloads/`, but not required.
1. **Fixture class** — use `FixtureBase` with benchmark-specific `PARAMS`, or `pytest.mark.parametrize` directly.
1. **Benchmark class** in `benchmarks/ops/bench_<op>.py` — subclass `BenchmarkBase`, implement `calculate_flops()` and `calculate_memory()` (return `None` if not applicable).
1. **Benchmark function** — `@YourFixture` decorated, construct workload + benchmark, call `inputs = workload.gen_inputs()`, then `bm.profile(op, *inputs)` and `BenchmarkReport.record(op, locals(), result, tag="tileops")`.
1. **Independent baseline** — record at least one non-`"tileops"` baseline (e.g., `"torch"`, `"fa3"`). If benchmark needs a ref function, define it locally — never import from `tests/` or `workloads/`.

### Metrics

- Latency (ms)
- TFLOPS (Tera Floating-point Operations Per Second)
- DRAM Bandwidth (GB/s)

### Reporting rules

- Numbers must come from a real GPU machine, not a sandbox.
- Include small, medium, and large representative shapes.
- Do not cherry-pick favorable shapes; report regressions as-is.
- Run the targeted correctness suite on the same GPU before reporting benchmark numbers.
- `BenchmarkReport.record()` first argument may be the Op instance or a string name; stay consistent within a given benchmark file.
- `calculate_flops()` and `calculate_memory()` should return numeric values when the metric is available; return `None` only if the metric is not applicable, in which case it will be omitted from the report.
- Every benchmark must record at least one non-`"tileops"` baseline. Use existing tags (`"baseline"`, `"torch"`, `"fa3"`, `"fla"`, `"triton"`) and avoid introducing ad-hoc tags without updating downstream consumers.
