# TileOPs Development Guide

This document outlines the software engineering standards, architecture, and development workflow for the TileOPs project. All contributors must adhere to these guidelines to ensure code quality, maintainability, and performance.

## 1. Architecture Overview

TileOPs follows a strict **2-Layer Hierarchical Architecture**. This separation of concerns ensures that hardware-specific optimizations (Kernels) are decoupled from user-facing APIs (Ops).

| Layer  |    Name    |   Analog    | Description                                                                                                                                  |
| :----: | :--------: | :---------: | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **L2** |   **Op**   | `torch.ops` | **Stateless Dispatcher**: Hardware-agnostic entry point. Dispatches to specific kernels. Compatible with **CUDA-Graph** & **torch.compile**. |
| **L1** | **Kernel** |  TileLang   | **Implementation**: Raw TileLang kernels optimized for specific hardware (e.g., Hopper, Ampere).                                             |

______________________________________________________________________

## 2. Development Workflow

Developing a new operator involves a bottom-up approach, moving from Kernel implementation to Op abstraction.

### Step 0: Create Tracking Issue

- **Action**: Create a new issue using the **"New Operator Request"** template.
- **Goal**: Define scope and track progress across the 2 layers.
- **Task Decomposition**: For new operators, **break down the checklist items into detailed sub-issues** (i.e., **Kernel Implementation**, **Op Implementation**, **Benchmark Results**). This allows new contributors to pick up smaller, well-defined tasks and submit smaller PRs.
- **Definition of Done**: The issue is closed when the operator is fully implemented and verified.

### Step 1: Kernel Implementation (L1)

- **Location**: `tileops/kernels/{operator_name}/`
- **Goal**: Implement the core logic using TileLang.
- **Docstrings**: Detailed description of arguments and return values.
- **Definition of Done**: The kernel compiles and runs correctly.

### Step 2: Op Definition & Verification (L2)

- **Location**: `tileops/ops/{operator_name}.py`
- **Responsibilities**:
  - Wrap the kernel in a Python function.
  - **Docstrings**: Google Style (Args, Returns, Example).
  - **Unit Test**: Compare output against a pure PyTorch reference implementation (required).
  - **Dtype Contract**: Explicitly define supported input dtypes, output dtype, and rejected dtypes.
  - **Benchmark**: Measure Latency, TFLOPS (required) and DRAM Bandwidth (required).
- **Standards**:
  - Use `torch.testing.assert_close` for floating-point verification.
    - **FP16**: `rtol=1e-3`, `atol=1e-3`
    - **BF16**: `rtol=1.6e-2`, `atol=1.6e-2`
  - Use exact comparison (`torch.equal`) for non-floating outputs such as `bool`, masks, and index tensors.
  - Tests must assert the output dtype when it differs from the input dtype.
  - GPU-dependent unit tests must be run on a real machine with host-visible CUDA devices. Do not treat sandbox-only results as final correctness evidence.
  - Benchmark results must be reproducible.
- **Definition of Done**: The op is verified in unit tests, and benchmarks run correctly.

### Step 3: Benchmark Results

- **Location**: `benchmarks/ops/bench_{operator_name}.py`
- **Goal**: Measure Latency, TFLOPS (required) and DRAM Bandwidth (required).
- **Execution**: `pytest benchmarks/` auto-generates `profile_run.log`.
- **Required Order**: Run the targeted correctness suite on the same real GPU machine before reporting benchmark numbers.
- **Required Shapes**: Benchmark tables must include representative small, medium, and large shapes unless the issue explicitly defines a different benchmark matrix.
- **Definition of Done**: Benchmark the op and put the results in the issue.

### Step 4: PR Acceptance Package

Every PR that adds a new op, expands dtype coverage, or changes semantic behavior must include the following in the PR description:

1. A dtype support matrix that states the actual supported input dtypes, output dtype, and PyTorch or documented baseline semantics.
1. A concrete acceptance checklist (`AC-1`, `AC-2`, ...) covering implementation, correctness tests, dtype contract checks, and benchmark deliverables.
1. A benchmark comparison table with real measured data and environment metadata.

Use the following benchmark table shape in the PR body:

| Shape / Params | dtype | Op         | TileOPs (ms) | Baseline (ms) | Ratio | Notes |
| -------------- | ----- | ---------- | ------------ | ------------- | ----- | ----- |
| example        | fp16  | example_op | ...          | ...           | ...   | ...   |

The PR body should stay concise: keep the summary, dtype matrix, acceptance checklist, and benchmark table. Do not paste long verification command transcripts into the PR description unless the reviewer explicitly asks for them.

If the implementation is correctness-first and performance follow-up is deferred, state that explicitly in the PR body and link the follow-up issue. Do not leave the benchmark section implicit.

______________________________________________________________________

## 3. Coding Standards

We enforce high standards for code quality and consistency.

### Python Code Style

- **Style Guide**: **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**.
  - We strictly follow the Google style for formatting and docstrings.
- **Formatter/Linter**: `ruff`
- **Docstrings**: All public functions and classes must use **Google-style docstrings**.

### Improvements & Type Safety

- **Type Hints**: All function signatures (inputs and outputs) must be type-hinted.
- **Strict Typing** *(planned)*: L2 (Op) APIs will be checked with `mypy` in strict mode in a future release.

______________________________________________________________________

## 4. Testing & Benchmarking Strategy

Tests and benchmarks are **separated by concern**: `pytest tests/` validates correctness only, `pytest benchmarks/` runs profiling only and auto-generates a markdown report (`profile_run.log`).

### Core Abstractions

| Class             | Location                  | Role                                                                                                                        |
| ----------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `FixtureBase`     | `tests/test_base.py`      | Metaclass-based `@decorator` that applies `pytest.mark.parametrize` from a `PARAMS` class attribute.                        |
| `TestBase`        | `tests/test_base.py`      | ABC with `gen_inputs()`, `ref_program()`, `check()`, `check_fn()`. Each op subclasses this.                                 |
| `BenchmarkBase`   | `benchmarks/benchmark.py` | ABC wrapping a `TestBase` instance. Subclass implements `calculate_flops()` and `calculate_memory()`. Provides `profile()`. |
| `BenchmarkReport` | `benchmarks/benchmark.py` | Static collector — `record()` stores results, `dump()` writes markdown, `clear()` resets.                                   |

### Pattern

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

### Unit Tests

- **Framework**: `pytest`
- **Location**: `tests/`
- **Requirement**:
  - Each op defines a `TestBase` subclass in `tests/ops/` with `gen_inputs()` and `ref_program()`.
  - Tests must cover `FP16` and `BF16` data types.
  - Tests must parameterize over common shapes (Batch size, Heads, Sequence length).
  - Tests must encode the dtype contract explicitly: supported dtypes are covered, unsupported dtypes are rejected, and output dtypes are asserted.
  - Changes to shared test infrastructure such as `tests/test_base.py`, common fixtures, or shared comparators must preserve existing default semantics unless the migration plan updates all affected tests in the same PR.
  - Before claiming the implementation is ready, run the full targeted test files for the affected op family on a real GPU machine, not just reject-path or smoke subsets.
  - If a PR touches shared test infrastructure, also run a broader real-machine `pytest -m smoke` pass before merge to catch regressions outside the target op family.

### Benchmarks

- **Framework**: `benchmarks.benchmark.BenchmarkBase`
- **Location**: `benchmarks/ops/`
- **Execution**: `pytest benchmarks/` — auto-generates `profile_run.log` (markdown format).
- **Metrics**:
  - Latency (ms)
  - TFLOPS (Terra Floating-point Operations Per Second)
  - DRAM Bandwidth (GB/s)
- **Reporting Rules**:
  - Benchmark numbers reported in PRs and issues must come from a real GPU machine, not a sandbox without direct CUDA visibility.
  - Report small, medium, and large representative shapes.
  - Do not cherry-pick only favorable shapes; if a representative large-shape result regresses versus baseline, report it as-is.

______________________________________________________________________

## 5. Directory Structure Reference

```text
TileOPs/
├── tileops/
│   ├── kernels/   # L1: TileLang Kernels
│   ├── ops/       # L2: OP + Dispatcher
│   └── utils/     # Utils
├── tests/         # Unit tests
├── benchmarks/    # Benchmarks and performance scripts
└── docs/          # Project documentation
```

## 6. Pull Request Process

### Before Submitting a PR

1. **Format Code**: Run pre-commit hooks to ensure code style compliance.
   ```bash
   pre-commit run --all-files
   ```
1. **Run Tests**: Ensure all relevant unit tests pass locally.
   ```bash
   PYTHONPATH="$PWD" python -m pytest tests/ops/test_<op_name>.py
   ```

### CI/CD Checks

When you open a PR, the following automated checks will run:

- **Lint**: Checks code style (Google Style), imports sorting, and spelling.
- **Test**: Runs unit tests and benchmarks on GPU runners.
- **Build**: Verifies the package builds successfully.

**Note**:

- Merging is blocked until all CI checks pass.
- **Approval**: Follow the **2+1 Policy** (2 Peers + 1 Mentor). See **[CONTRIBUTING.md](./CONTRIBUTING.md)**.
