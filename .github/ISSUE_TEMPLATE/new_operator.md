---
name: New Operator Request
about: Track the development of a new operator
title: '[New Op] <Operator Name>'
labels: enhancement, operator
assignees: ''
---

## Operator Description

<!-- detailed description of the operator, including mathematical formula if applicable -->

## Implementation Plan

### 1. Kernel Implementation (L1)

- [ ] **Kernel**: Implement TileLang kernel in `top/kernels/<op_name>/`
- [ ] **Verification**: Pass functional correctness tests

### 2. Op Definition (L2)

- [ ] **Interface**: Define `torch.ops` interface in `top/ops/<op_name>.py`
- [ ] **Unit Tests**: Implement `tests/test_<op_name>.py` (Compare vs PyTorch Ref)
  - [ ] FP16 (close: 1e-3)
  - [ ] BF16 (close: 1.6e-2)
- [ ] **Benchmarks**: Implement `benchmarks/benchmark_<op_name>.py`
  - [ ] Latency
  - [ ] TFLOPS
  - [ ] DRAM Bandwidth

### 3. Functional API (L3)

- [ ] **Autograd**: Implement `torch.autograd.Function` in `top/functions/<op_name>.py`
- [ ] **GradCheck**: Pass `torch.autograd.gradcheck`

### 4. Layer Wrapper (L4)

- [ ] **Module**: Implement `nn.Module` in `top/layers/<op_name>.py`
- [ ] **Type Hints**: Complete type hints (strict mode checked)

### 5. Benchmark Results

<!-- Please report the benchmark results in the table below -->

|       Shape        | FP16 Latency (ms) | FP16 TFLOPS | FP16 DRAM Bandwidth (GB/s) | BF16 Latency (ms) | BF16 TFLOPS | BF16 DRAM Bandwidth (GB/s) |
| :----------------: | :---------------: | :---------: | :------------------------: | :---------------: | :---------: | :------------------------: |
| (B=1, S=1024, ...) |        ...        |     ...     |            ...             |        ...        |     ...     |            ...             |

## Reference

<!-- Links to papers, pytorch docs, or reference implementations -->
