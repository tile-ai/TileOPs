---
name: New Operator Request
about: Track the development of a new operator
title: '[New Op] <Operator Name>'
labels: enhancement, operator
assignees: ''
---

## Operator Description

<!-- detailed description of the operator, including mathematical formula if applicable -->

## Dtype Support Matrix

<!-- List only the dtypes this operator will actually support in the first PR. -->

| Op / API   | Input dtypes     | Output dtype  | PyTorch / reference semantic baseline |
| ---------- | ---------------- | ------------- | ------------------------------------- |
| \<op_name> | fp16, bf16, fp32 | same as input | torch.\<op_name>                      |

## Implementation Plan

### 1. Kernel Implementation (L1)

- [ ] **Kernel**: Implement TileLang kernel in `tileops/kernels/<op_name>/`
- [ ] **Verification**: Pass functional correctness tests

### 2. Op Definition (L2)

- [ ] **Interface**: Define `torch.ops` interface in `tileops/ops/<op_name>.py`
- [ ] **Unit Tests**: Implement `tests/test_<op_name>.py` (Compare vs PyTorch Ref)
  - [ ] FP16 (close: 1e-3)
  - [ ] BF16 (close: 1.6e-2)
  - [ ] Output dtype matches the declared contract
  - [ ] Non-floating exact outputs (`bool`, indices, masks) use exact comparison (`torch.equal`)
- [ ] **Benchmarks**: Implement `benchmarks/benchmark_<op_name>.py`
  - [ ] Latency
  - [ ] TFLOPS
  - [ ] DRAM Bandwidth

## Acceptance Criteria

- [ ] AC-1:
- [ ] AC-2:
- [ ] AC-3:

### 3. Benchmark Results

<!-- Please report the benchmark results in the table below -->

**Configuration**: <!-- GPU, CUDA, torch -->

| Shape / Params     | dtype | Op         | TileOPs (ms) | Baseline (ms) | Ratio | TFLOPS / BW | Notes |
| ------------------ | ----- | ---------- | ------------ | ------------- | ----- | ----------- | ----- |
| (B=1, S=1024, ...) | fp16  | \<op_name> | ...          | ...           | ...   | ...         | ...   |

## Reference

<!-- Links to papers, pytorch docs, or reference implementations -->
