---
name: New Operator Sub-task
about: A specific sub-task for implementing a part of a new operator
title: '[<Feat/Perf/Fix>][Ops] <Operator Name>: <manifest/test/impl/bench scope>'
labels: sub-task, operator
assignees: ''
---

## Parent Issue

<!-- Link to the main tracking issue for this operator using #IssueID -->

Part of #

## Task Type

<!-- Please check the relevant component for this sub-issue -->

- [ ] **Manifest / Spec** (required for any public Op)
- [ ] **Kernel / Op Implementation**
- [ ] **Correctness Tests / Workloads**
- [ ] **Benchmark / Performance**

## Description

<!-- Detailed description of what needs to be implemented in this step -->

## Checklist

<!--
Refer to:
- docs/design/manifest.md for manifest fields and validation
- docs/design/ops-design.md for Op/Kernel implementation rules
- docs/design/testing.md for tests, workloads, and benchmark structure
- docs/design/roofline.md for roofline authoring and benchmark consumers
-->

- [ ] Public Op has a `src/tileops/manifest/` entry (or updates an existing entry).
- [ ] Manifest `signature` declares indices, inputs, outputs, params and refinements.
- [ ] Manifest `workloads` rows declare the benchmark calls and their `dtype_cases`; unit-test edge cases are not generated from them.
- [ ] Manifest `roofline` is present and consumable by `op.eval_roofline()`.
- [ ] Op constructor and `forward()` match the manifest entry; `kernel_types` lists the kernels the op dispatches.
- [ ] Tests use an independent reference implementation and cover relevant FP16/BF16 and edge cases.
- [ ] Benchmarks take their calls from `manifest_calls` and record at least one non-`tileops` baseline unless explicitly justified.
- [ ] PR title and commits follow the current `[Feat][Scope]` / `[Perf][Scope]` / `[Fix][Scope]` convention.
- [ ] Implementation follows **Google Python Style** for code and docstrings.
