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

- [ ] `pre-commit run --all-files` passed
- [ ] Relevant local unit tests passed
- [ ] **(New ops)** Structural compliance verified (see below)
- [ ] **(New ops)** Dtype support matrix documented above
- [ ] **(New ops)** Benchmark results reported above and in tracking issue
