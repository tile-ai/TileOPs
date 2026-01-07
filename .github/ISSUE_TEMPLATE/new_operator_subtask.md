---
name: New Operator Sub-task
about: A specific sub-task for implementing a part of a new operator
title: '[New Op Sub-task] <Operator Name> - <L1/L2/L3/L4/Benchmark>'
labels: sub-task, operator
assignees: ''
---

## Parent Issue

<!-- Link to the main tracking issue for this operator using #IssueID -->

Part of #

## Task Type

<!-- Please check the relevant component for this sub-issue -->

- [ ] **L1: Kernel Implementation** (Write TileLang kernel)
- [ ] **L2: Op Implementation** (Wrapper + Unit Tests + Benchmarks)
- [ ] **L3: Function Implementation** (Autograd Function)
- [ ] **L4: Layer Implementation** (nn.Module Wrapper)
- [ ] **Benchmarks** (Performance Profiling)

## Description

<!-- Detailed description of what needs to be implemented in this step -->

## Checklist

<!-- Refer to docs/DEVELOPMENT.md for specific requirements for each layer -->

- [ ] Implementation follows **Google Python Style** for code and docstrings.
- [ ] **(L1 Only)** Kernel verified on unit tests.
- [ ] **(L2 Only)** Unit tests match PyTorch reference (FP16/BF16).
- [ ] **(L2 Only)** Benchmarks implemented (Latency/TFLOPS/Bandwidth).
- [ ] **(L3 Only)** GradCheck passed.
- [ ] **(L4 Only)** Verified as `nn.Module`.
