# Trust Model

Each development stage owns a specific concern. Boundaries prevent one stage from silently weakening another's guarantees.

## Pipeline

```
Manifest → Test → Implementation → Benchmark
```

Each stage declares its trust contract using a fixed heading taxonomy:

- **OWNS** — write authority: the artifacts (data, files, content categories) this stage is responsible for authoring. Required on every stage.
- **MUST NOT WRITE** — write restrictions: artifact or content categories this stage must not author, to prevent corrupting another stage's invariants. Required on every stage. Items here name content scope, not raw filesystem mutation — e.g. "the manifest stage must not author kernel internals" applies whether the proposed prose lives in a manifest file, a docstring, or anywhere else this stage might write.
- **MUST NOT** — structural coupling restriction: a cross-stage linkage rule that is not about writes — typically a forbidden import that would collapse independent verification surfaces (e.g. benchmark importing tests' oracle/ref code). Optional — appears only on stages that have such a rule (currently only Benchmark).

Mere reading is not policed by the trust model: any stage may read any file when its job calls for it. The `MUST NOT` heading does not forbid reading per se; it forbids specific structural couplings — typically import-level dependencies — that would defeat the purpose of having independent stages. Per-stage rules live in each [domain rule file](../../.claude/domain-rules/).

## Manifest

Source of truth for op interfaces. Human-reviewed, separate PR.

- **OWNS**: op signatures, dtypes, workload shapes, roofline formulas, status, kernel_map (dispatch registration table)
- **MUST NOT WRITE**: kernel internals, dispatch strategy, or test logic

→ Rules: [manifest-spec.md](../../.claude/domain-rules/manifest-spec.md) | Guide: [manifest.md](manifest.md)

## Test

PR-level correctness verification. QA writes tests against manifest spec.

- **OWNS**: ref_program, tolerances, assertions, `tests/`, `workloads/`
- **MUST NOT WRITE**: kernel code, benchmark logic, or performance measurements

→ Rules: [testing-budget.md](../../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Tests](testing.md#tests)

## Implementation

Kernel (L1) + Op (L2). Developer reads manifest + ref_program for behavior; high-perf optimization is independent.

- **OWNS**: TileLang kernels, op dispatch, class variable protocol
- **MUST NOT WRITE**: workload shape definitions, correctness assertions, manifest entries

→ Rules: [ops-design.md](../../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md)

## Benchmark

Nightly performance guard. Independent baselines — cannot modify op/tests/workloads.

- **OWNS**: profiling, baseline comparisons, `benchmarks/`
- **MUST NOT WRITE**: correctness assertions, kernel code
- **MUST NOT** (oracle-leakage rule, not a write rule): import oracle/ref functions from `tests/` or `workloads/`. Benchmark-local baseline functions are allowed; the prohibition prevents the benchmark stage from sharing its correctness oracle with the test stage.

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads Layer

Shared input-definition layer — not a development stage. Test stage OWNS it (QA creates workload classes first).

**Provides**: `WorkloadBase` (gen_inputs), `FixtureMeta`/`FixtureBase` (parametrize), per-op workload subclasses.

**Must not contain**: ref_program, check/tolerance logic, calculate_flops/memory, benchmark baselines. Reason: prevents shared oracle surface between test correctness and benchmark baselines.

```
WorkloadBase (workloads/workload_base.py)  # gen_inputs() only — default implementation
  ├── TestBase (tests/test_base.py)     # adds ref_program(), check()
  └── concrete subclasses typically define shape + dtype

# Public benchmark interface (capability protocols)
ShapeDtypeWorkload                      # shape + dtype metadata
InputGeneratingWorkload                 # gen_inputs()
BenchmarkWorkload                       # both (when a workload defines shape, dtype, gen_inputs)
BenchmarkBase[W] (benchmarks/)          # generic over workload type
```

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
