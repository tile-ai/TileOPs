# Development Workflow

Bottom-up development flow: Kernel → Op → Test → Benchmark → PR.

## Step 0: Create Tracking Issue

- Create a new issue using the **"New Operator Request"** template.
- Define scope and track progress across the 2 layers.
- Break checklist items into sub-issues (Kernel, Op, Benchmark) for smaller PRs.
- Issue is closed when the operator is fully implemented and verified.

## Step 1: Kernel Implementation (L1)

- **Location**: `tileops/kernels/{operator_name}/`
- Implement core logic using TileLang.
- Detailed docstrings for arguments and return values.
- **Done when**: kernel compiles and runs correctly.

## Step 2: Op Definition and Verification (L2)

- **Location**: `tileops/ops/{operator_name}.py`
- Wrap the kernel in a Python function.
- Google-style docstrings (Args, Returns, Example).
- Unit test comparing output against a pure PyTorch reference.
- Dtype contract: explicitly define supported input dtypes, output dtype, rejected dtypes.
- Parameter contract: validate scalar parameters at the Op boundary. Invalid values fail with `ValueError` before any TIR/codegen step.
- Do not fix interface-contract bugs with kernel-local workarounds. Fix the validation boundary first.
- **Done when**: op passes unit tests.

## Step 3: Benchmark

- **Location**: `benchmarks/ops/bench_{operator_name}.py`
- Measure latency, TFLOPS, and DRAM bandwidth.
- Run correctness suite on the same GPU before reporting numbers.
- Include small, medium, and large representative shapes.
- **Done when**: benchmark results are posted in the tracking issue.

See [testing.md](testing.md) for framework details, tolerances, and reporting rules.

## Step 4: PR Submission

### PR Acceptance Package

Every PR that adds a new op, expands dtype coverage, or changes semantic behavior must include:

1. **Dtype support matrix** — supported input dtypes, output dtype, baseline semantics.
1. **Acceptance checklist** (`AC-1`, `AC-2`, ...) — implementation, correctness tests, dtype contract, benchmarks.
1. **Benchmark comparison table** with real measured data and environment metadata:

| Shape / Params | dtype | Op         | TileOPs (ms) | Baseline (ms) | Ratio | Notes |
| -------------- | ----- | ---------- | ------------ | ------------- | ----- | ----- |
| example        | fp16  | example_op | ...          | ...           | ...   | ...   |

Keep the PR body concise: summary, dtype matrix, acceptance checklist, benchmark table. Do not paste long verification transcripts unless the reviewer asks.

If performance work is deferred, state that explicitly and link the follow-up issue.

### Before Submitting

1. Run pre-commit hooks:
   ```bash
   pre-commit run --all-files
   ```
1. Run relevant unit tests:
   ```bash
   PYTHONPATH="$PWD" python -m pytest tests/ops/test_<op_name>.py
   ```

### CI/CD Checks

PRs trigger automated checks:

- **Lint**: code style (Google Style), import sorting, spelling.
- **Test**: unit tests and benchmarks on GPU runners.
- **Build**: package build verification.

Merging is blocked until all CI checks pass.

## Branch and Commit Conventions

### Branch Naming

Format: `type/scope/description`. Canonical type prefixes are in `.claude/conventions/types.sh`.

Examples:

- `feat/flash-attn/fwd-kernel`
- `fix/mla/parsing-error`
- `doc/readme/add-examples`
- `test/gemv/add-edge-cases`
- `perf/mha/improve-bandwidth`
- `bench/gemm/add-triton-baseline`

### Commit Messages

Format: `[Type] Description` or `[Type][Scope] Description`.

| Type            | Usage                                      |
| --------------- | ------------------------------------------ |
| `[Feat]`        | New features or operators                  |
| `[BugFix]`      | Bug fixes                                  |
| `[Fix]`         | Non-bug corrections                        |
| `[Refactor]`    | Code restructuring without behavior change |
| `[Enhancement]` | Improvements to existing features          |
| `[Doc]`         | Documentation updates                      |
| `[Chore]`       | Build system or workflow changes           |
| `[Bench]`       | Benchmark updates                          |
| `[CI]`          | CI/CD changes                              |
| `[Test]`        | Test changes                               |
| `[Perf]`        | Performance improvements                   |

Examples:

- `[Feat] Add multi-head attention forward op`
- `[BugFix] Fix index out of bounds in reduction kernel`
- `[Enhancement][MHA] Improve multi-head attention forward op performance on Hopper`

Keep commit messages concise. Long verification sections, benchmark tables, and command transcripts belong in the PR description, not commit bodies.

## Coding Standards

- **Style**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- **Formatter/Linter**: `ruff`.
- **Docstrings**: Google-style on all public functions and classes. Type info goes in Python type hints, not repeated in docstrings.
- **Type Hints**: all function signatures (inputs and outputs) must be type-hinted.
- **Strict Typing** *(planned)*: L2 (Op) APIs will be checked with `mypy` in strict mode.
