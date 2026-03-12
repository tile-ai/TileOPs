## Description

<!-- Briefly describe the changes introduced by this PR -->

## Dtype Support Matrix

<!-- Required for new ops or dtype-contract changes. State the actual supported set, not the aspirational set. -->

| Op / API   | Input dtypes     | Output dtype  | Reference baseline |
| ---------- | ---------------- | ------------- | ------------------ |
| example_op | fp16, bf16, fp32 | same as input | torch.example_op   |

## Test Plan

<!-- For new ops, enumerate acceptance criteria as AC-1, AC-2, ... and mark each item with concrete evidence. -->

- [ ] AC-1:
- [ ] AC-2:
- [ ] AC-3:

## Benchmark

<!-- Required for new ops and performance-sensitive semantic changes. Use real measured data from a host-visible CUDA machine and compare against a baseline. -->

**Configuration**: <!-- GPU, CUDA, torch -->

| Shape tier | Shape / Params | dtype | Op         | TileOPs (ms) | Baseline (ms) | Ratio | Notes |
| ---------- | -------------- | ----- | ---------- | ------------ | ------------- | ----- | ----- |
| small      | example        | fp16  | example_op | ...          | ...           | ...   | ...   |
| medium     | example        | fp16  | example_op | ...          | ...           | ...   | ...   |
| large      | example        | fp16  | example_op | ...          | ...           | ...   | ...   |

<!-- Do not cherry-pick only favorable shapes. If a representative large-shape result regresses, report it as-is. -->

<!-- If the implementation is correctness-first or a benchmark is intentionally deferred, say so explicitly and link the follow-up issue. -->

## Type of Change

- [ ] Bug fix
- [ ] New operator implementation
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Infrastructure/CI

## Checklist

- [ ] I have run `pre-commit run --all-files` and fixed all linting issues.
- [ ] I have verified that my changes pass the relevant local unit tests.
- [ ] **(For new ops)** I have documented the supported input dtypes, output dtype, and baseline semantic reference in the PR body.
- [ ] **(For new ops)** I have listed concrete acceptance criteria in the PR body and verified each one.
- [ ] **(For new ops)** I have added the corresponding `Benchmark` class in `benchmarks/`.
- [ ] **(For new ops)** I have run GPU-dependent tests and benchmarks on a real CUDA-visible machine.
- [ ] **(For new ops)** I have reported measured benchmark results for small, medium, and large representative shapes in the PR body and tracking issue.
