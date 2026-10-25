# Testing and Benchmarking

`pytest tests/` checks correctness. `pytest benchmarks/` checks the same correctness contract, then times each case and writes `profile_run.log`.

## Core Abstractions

| Class             | Role                                                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `WorkloadBase`    | Declares `gen_inputs()` and `verification(*inputs)`. A concrete operator workload adds `ref_program()` and owns numerical policy. |
| `FixtureBase`     | Applies `pytest.mark.parametrize` from a `PARAMS` attribute or a `get_params()` classmethod.                                      |
| `TestBase`        | Adds `check()`, which runs the shared verifier under pytest.                                                                      |
| `bench.Runner`    | Verifies, times and records the TileOps op and the implementations compared with it on one `bench.Case`.                          |
| `BenchmarkReport` | Collects every row and writes the report.                                                                                         |

## Wiring

A workload is defined once. A test and a benchmark each use it, and never each other.

| Layer                         | Holds                                                                                             |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| Workload (`workloads/`)       | `ref_program()`, `verification(*inputs)`, and the input construction the manifest rows do not fix |
| Test (`tests/ops/`)           | `(Workload, TestBase)`; no reference and no numerical override                                    |
| Benchmark (`benchmarks/ops/`) | `bench.Runner(op, case).compare({...})` over `bench.cases(Op)`                                    |

- A benchmark imports workloads, never tests.
- The reference lives on the narrowest class that names one operator, even when its base describes only an input shape.
- Every call a benchmark publishes comes from a manifest workload row and its `dtype_cases`. Fixture parameters only set controls that change no call.
- [`tests/test_workload_placement.py`](../../tests/test_workload_placement.py) checks the split: references and verification declarations live in workloads, and a workload holds no pytest check or roofline calculation.

## Tests

→ Boundary: [layer-boundaries.md §Test](layer-boundaries.md#test)

**Framework:** pytest. A test function decorated by its fixture calls `test.check(op, *test.gen_inputs())`.

**Contract:** the ops are the external interface. A test guards what an op call promises for the inputs and parameters a caller can pass, tuned mode (`request_tune()`) included. A direct kernel call or a config handed to an in-tree kernel is promised nothing, and no test exercises it. The exception is a call-protocol test: it may publish a configuration on a kernel the op reached, because publication is what it tests ([ops-design.md § Call protocol](ops-design.md#call-protocol)).

**Location:**

- [`tests/ops/`](../../tests/ops/): every test of an op's behavior. It reaches the kernel through the op's dispatch, the only path a caller has.
- [`tests/kernels/`](../../tests/kernels/): only a shared mechanism no single kernel owns that a promise depends on, such as an autotune sweep helper.

**Reachability:**

- A test reaches a path with an input that dispatch sends there. A path no input reaches on any device is a dispatch defect: dispatch is fixed, or the code is removed. A stand-in implementation a test adds through `register_kernel_type` tests registration only.
- A path another device selects is tested on that device; a test does not emulate a device.
- A test of such a path carries that architecture's marker and is skipped elsewhere. Its docstring names the hardware fact the path depends on, never a byte count or threshold of the current kernel.
- Autotune candidates are covered by one tuned-mode test per op, calling `request_tune()` on the constructed op, that checks the chosen result.
- An input inside the signature domain that every implementation refuses is a coverage gap; one test per gap asserts the op refuses it before building anything and names the reason.
- `check(runs=...)` takes only a compiled or wrapped form of the op; it refuses a kernel.

**Target:**

- A run defaults to `BUILTIN`, so an installed backend does not serve it. `--tileops-target=detect` restores device detection; `--tileops-target=<name>` selects a target.
- A test of target dispatch names its target or isolates the registry.
- An op contract test asserts nothing about the in-tree implementation — the kernel class, strategy or config chosen, or how a cache key folds a shape.
- A test that depends on the in-tree implementations — a coverage gap, a registered stand-in implementation, a kernel path it pins `target=BUILTIN` to reach — carries `in_tree_kernels`, which deselects it off the builtin target.
- The full `tests/ops` suite on a device other than CUDA runs under a `--tileops-target` naming a backend that serves the device; `cpu` with the builtin target is not a configuration of that suite.

**Device:**

- Tensors go on `workloads.device.run_device()`, the device `--tileops-device` names (default `cuda`).
- A device query or control goes through `torch.accelerator` where it has an equivalent, so the test runs on any accelerator.
- A test that needs CUDA in a run configuration that collects it carries `cuda_only`, writes `"cuda"`, and is deselected on any other device. It needs CUDA when it, for example:
  - asserts a CUDA device fact of a call record;
  - needs a second CUDA device;
  - calls a `torch.cuda` API with no `torch.accelerator` equivalent;
  - checks against a reference that runs only on CUDA;
  - builds or calls a CUDA kernel directly, outside an op;
  - executes an in-tree CUDA kernel outside `tests/ops`, where a `cpu` run uses the builtin target.
- In `tests/ops`, a test that needs CUDA only because the in-tree kernels do carries `in_tree_kernels` alone: off CUDA that suite runs under a backend target, which deselects it.
- An `sm*`-marked case needs CUDA through its architecture mark and does not also carry `cuda_only`.
- A test that puts a run-device tensor beside a CPU one skips where the run device is `cpu`; it does not carry `cuda_only`.
- An availability gate asks `workloads.device.run_device_available()`, not `torch.cuda.is_available()`.

### Shared correctness contract

- A concrete workload's `verification(*inputs)` returns an `Exact`, `Partial` or `Custom` declaration from `workloads.numerics`. Tests and every benchmark tag share its reference, output coverage, structural checks (shape, dtype, device) and tolerance.
- `ref_program` computes a float result at float32 or wider, never below the output's precision, and rounds once to the output dtype. It rounds an intermediate only where the op's contract stores or documents it. Integer, boolean and data-movement results are exact at their own dtype.
- `check()` and `compare()` take no comparator, tolerance or evidence override. Operator semantics change `verification()`; only a protocol change touches the shared verifier.
- A reference out of memory records no verification. A subject failure propagates.
- `Partial` names what stays unchecked, and missing evidence never yields a ratio. JUnit records `checked_outputs`, even when the error is zero.
- Framework self-tests call the shared verifier directly and impersonate no operator.
- An ordinary call checks structure, effects and storage ([ops-design.md § Call protocol](ops-design.md#call-protocol)), never numerics; numerical correctness is checked here, through the shared verifier.
- A call-protocol test covers a cold call, a repeat, a changed dependency and a return to an earlier specialization, and fails when the step it guards is removed.

### Tolerance

The default bound is per output dtype, from `reference_tolerance(dtype)`:

| Dtype | `rtol` | `atol` |
| ----- | ------ | ------ |
| FP32  | 1e-5   | 1e-5   |
| FP16  | 1e-3   | 1e-3   |
| BF16  | 1.6e-2 | 1.6e-2 |

- Non-floating outputs (bool, masks, indices) compare exactly.
- A workload declares another bound in `verification()` only when its semantics need one, and measures before loosening.
- `atol` follows summation order, not reduction length. A kernel that sums in another order than the reference (GEMV, cross-thread allreduce, split-K) leaves cancelled outputs that only `atol` covers; an order-matching kernel (dense GEMM, BMM against cuBLAS) is bit-exact.

### Coverage rules

- Tests cover the dtype domain the signature declares: supported dtypes pass, unsupported ones are rejected, and an output dtype that differs from the input is asserted.
- GPU results come from a real machine with CUDA devices. A sandbox result is not correctness evidence.
- CUDA Graph support is qualified per public call by verifying outputs and observable state against the shared workload after replay with updated inputs.
- Benchmark coverage requires correctness evidence for every declared implemented-op call exactly once.

### Test case policy

Each parameterized case serves exactly one purpose:

1. **Dtype correctness**: a supported dtype.
1. **Shape coverage**: a distinct code path (boundary, tile edge, alignment).
1. **Feature coverage**: a flag or mode (`causal=True`, tuned mode through `request_tune()`).
1. **Regression**: a fixed bug. The docstring states the fault it guards; an issue or PR number does not belong in shipped source (`scripts/lint/shipped_refs_lint.py`).

No performance exploration, autotune sweep or duplicate code path.

**Tier:** every case carries exactly one of `smoke`, `full` and `nightly`; collection fails otherwise.

**Dtype:** every supported dtype. Smoke covers each with one typical shape; full adds cross-combinations only where each guards a named code path.

**Shape:** the smallest shape that triggers each kernel branch. These cases are separate from the manifest contract cases, which come from the entry's workload rows. Typical branches:

- Tile boundary: a shape the tile does not divide.
- Vector alignment: a shape off the vector width.
- Degenerate dimension: a size of 1.
- Dispatch range: shapes that select different kernels.
- Broadcast, for a binary op (arithmetic, comparison, logical, bitwise): bias-add `(B, S, D) + (1, 1, D)`, row `(B, S, D) + (B, S, 1)` and scalar `(M, N) + (1, 1)`.

**Growth:**

- Each new case states its purpose in a comment or the PR.
- A function with over 20 cases justifies the count by code path.
- Different behavior gets a new function rather than more cases.
- [`scripts/test_node_delta.py`](../../scripts/test_node_delta.py) compares collected nodes against the base branch and never fails.

### Testing layers

| Layer             | Responsibility                                      | Shape source                                                     |
| ----------------- | --------------------------------------------------- | ---------------------------------------------------------------- |
| UT smoke/full     | Guard PR correctness                                | The implementer, by kernel code path                             |
| Nightly benchmark | Performance regression + typical/stress correctness | [`src/tileops/manifest/`](../../src/tileops/manifest/) workloads |
| Local dev         | Performance tuning verification                     | The developer, ad hoc                                            |

### Infrastructure rules

- A change to shared test infrastructure (`tests/workload_test_base.py`, common fixtures, the shared verifier) keeps default semantics unless every affected test migrates with it.

## Benchmarks

→ Boundary: [layer-boundaries.md §Benchmark](layer-boundaries.md#benchmark)

**Location:** [`benchmarks/ops/`](../../benchmarks/ops/), one `bench_<op>.py` per op module with its variants (inference, decode, paged, end-to-end). `pytest benchmarks/` writes `profile_run.log`.

**Target and device:**

- The repository-root `conftest.py` owns `--tileops-target` and `--tileops-device` for tests and benchmarks alike. A benchmark run defaults to `BUILTIN`; a backend measuring itself names its target.
- A benchmark runs on CUDA or on the process's accelerator, and times every implementation in a row on that device.
- A row records how it was timed: CUPTI on CUDA, CUDA events where CUPTI fails and the run allows the fallback, device events on another accelerator. Events cannot separate execution from gaps, so an event-timed row's `device_busy_ms` equals its `latency_ms`.
- A baseline library unavailable on the run device skips a row that selects it and drops the tag of one that prefers it. On CUDA, a missing selected library fails the row. Why: the runner image installs every selected baseline.
- `torch-compile` is part of torch, not a selected library. Off CUDA, on a device Inductor cannot compile for (no Inductor backend, or no Triton), `Runner.compare` leaves the tag out with a pytest warning and records the rest of the row. On CUDA, a torch-compile failure fails the row.

### File checklist

1. **Cases**: parametrize over `bench.cases(Op)`. Register the op's workload in [`benchmarks/_cases/`](../../benchmarks/_cases/). Benchmark files use the resulting `Case` and do not construct inputs or read the manifest `Call`.
1. **Comparison**: construct the op from `case.arguments`, then call `bench.Runner(op, case).compare(...)`. Include an independent implementation; the runner adds `tileops`, verifies the comparison and records the report.
1. **Baselines**: name each implementation in lowercase, hyphen-separated. A torch baseline is `case.torch`, the workload's `torch_program` at the input dtype; a workload without one falls back to its reference. A semantic difference belongs to `Implementation.noncomparable_reason` and has no ratio. Resolve library implementations through [`benchmarks/baselines.py`](../../benchmarks/baselines.py): a selected library through a `*_op` resolver or `baseline_module`, a preferred one through `optional_baseline`, whose `None` drops the tag. A library that cannot express the case has no tag, and the file states why. Never import from `tests/`.
1. **State**: each implementation preserves shared case inputs. Put overwritten arguments in private storage and restore them in `reset`; each round runs `reset`, the L2 flush, then the timed call.
1. **Copies**: `count_copies` belongs to the case's registry entry and applies to every implementation in the comparison.
1. **Launch**: a timed callable launches its own work. Gradients are driven through `backward_of`: `Tensor.backward` launches kernels on another thread, outside the timer's attribution.

### Verification

- Every implementation is checked against one run of the reference before any is timed. Reference results are released and inputs restored before sampling.
- An implementation's adapter returns the workload's output structure, shape and dtype; no implementation carries its own tolerance or reference.
- A benchmark asserts what it needs to trust a measurement, not the op's behavioral contract.
- A torch-compile row uses `compiled_reference`, which requires one graph with no break.
- `--tileops-verify` is a diagnostic mode, not a second nightly sweep.
- A row with no applicable reference carries an explicit verification gap and no ratio.

### Metrics

Latency (ms), TFLOPS, and DRAM bandwidth (TB/s).

### Reporting rules

- Numbers come from a real GPU, after the targeted correctness suite has passed on the same GPU.
- Report small, medium and large representative shapes. Do not cherry-pick; report regressions as they are.
- Each row is one op's measurement: `BenchmarkReport.record()` takes the op the runner was built with. A comparison of anything else — kernel strategies, a field of libraries — asserts.
- Use an existing baseline tag. A new tag means updating its downstream consumers.
