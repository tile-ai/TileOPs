# Trust Model

Each development stage owns a specific concern. Boundaries prevent one stage from silently weakening another's guarantees.

## Pipeline

```
Manifest → Test → Implementation → Benchmark
```

Each stage declares **OWNS** / **MUST NOT** / **MAY READ** in its [domain rule file](../.claude/domain-rules/).

## Manifest

Source of truth for op interfaces. Human-reviewed, separate PR.

- **OWNS**: op signatures, dtypes, workload shapes, roofline formulas, status, kernel_map (dispatch registration table)
- **MUST NOT**: contain kernel internals, dispatch strategy, or test logic
- **MAY READ**: PyTorch public API (to match signatures)

→ Rules: [manifest-spec.md](../.claude/domain-rules/manifest-spec.md) | Guide: [testing.md §Writing a Test](testing.md#writing-a-test)

## Test

PR-level correctness verification. QA writes tests against manifest spec.

- **OWNS**: ref_program, tolerances, assertions, `tests/`, `workloads/`
- **MUST NOT**: contain kernel code, benchmark logic, or performance measurements
- **MAY READ**: manifest (to verify interface), `workloads/` (via WorkloadBase inheritance)

→ Rules: [testing-budget.md](../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Writing a Test](testing.md#writing-a-test)

## Implementation

Kernel (L1) + Op (L2). Developer reads manifest + ref_program for behavior; high-perf optimization is independent.

- **OWNS**: TileLang kernels, op dispatch, class variable protocol
- **MUST NOT**: define workload shapes, own correctness assertions, modify manifest
- **MAY READ**: manifest (interface), `tests/` (behavior understanding — not to reverse-engineer passing)

→ Rules: [ops-design.md](../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md), [testing.md §Writing an Op](testing.md#writing-an-op-implementation)

## Benchmark

Nightly performance guard. Independent baselines — cannot modify op/tests/workloads.

- **OWNS**: profiling, baseline comparisons, `benchmarks/`
- **MUST NOT**: contain correctness assertions, kernel code, or import oracle/ref functions from `tests/` or `workloads/` (benchmark-local baseline functions are allowed)
- **MAY READ**: `workloads/` (composition), `tileops/ops/` (to profile)

→ Rules: [benchmark.md](../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Writing a Benchmark](testing.md#writing-a-benchmark)

## Workloads Layer

Shared input-definition layer — not a development stage. Test stage OWNS it (QA creates workload classes first).

**Provides**: `WorkloadBase` (gen_inputs), `FixtureMeta`/`FixtureBase` (parametrize), per-op workload subclasses.

**Must not contain**: ref_program, check/tolerance logic, calculate_flops/memory, benchmark baselines. Reason: prevents shared oracle surface between test correctness and benchmark baselines.

```
WorkloadBase (workloads/base.py)        # gen_inputs() only
  ├── TestBase (tests/test_base.py)     # adds ref_program(), check()
  └── BenchmarkBase (benchmarks/)       # composes WorkloadBase, adds profiling
```

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
