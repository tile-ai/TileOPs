# Trust Model

Each development stage owns a specific concern. Boundaries prevent one stage from silently weakening another's guarantees.

## Pipeline

```
Manifest → Test → Implementation → Benchmark
```

Each stage declares its trust contract using these headings:

- **OWNS** — what this stage authors. Required.
- **MUST NOT WRITE** — content this stage must not author, in any file. Required.
- **MUST NOT** — structural couplings (typically forbidden imports) that would defeat stage independence. Optional.

Reads are not policed; the trust model controls writes and import-level coupling, not file access. Per-stage rules live in each [domain rule file](../../.claude/domain-rules/).

## Manifest

Source of truth for op interfaces. Human-reviewed, separate PR.

- **OWNS**: op signatures, dtypes, workload shapes, roofline formulas, status, kernel_map (dispatch registration table)
- **MUST NOT WRITE**: kernel internals, dispatch strategy, or test logic

### Status flip carve-out

An implementation PR may edit only `status`, `source.kernel_map`, and (only when promoting `spec-only → implemented`) `workloads` on the aligned op; every other contractual field needs a separate manifest-only PR.

Full enumeration: [.claude/rules/manifest-trust-model.md](../../.claude/rules/manifest-trust-model.md) §Status flip carve-out.

→ Rules: [manifest-spec.md](../../.claude/domain-rules/manifest-spec.md) | Guide: [manifest.md](manifest.md)

## Test

PR-level correctness verification. QA writes tests against manifest spec.

- **OWNS**: ref_program, tolerances, assertions, [`tests/`](../../tests/), [`workloads/`](../../workloads/)
- **MUST NOT WRITE**: kernel code, benchmark logic, or performance measurements

→ Rules: [testing-budget.md](../../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Tests](testing.md#tests)

## Implementation

Kernel (L1) + Op (L2). Developer reads manifest + ref_program for behavior; high-perf optimization is independent.

- **OWNS**: TileLang kernels, op dispatch, class variable protocol
- **MUST NOT WRITE**: workload shape definitions, correctness assertions, manifest entries

→ Rules: [ops-design.md](../../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md)

## Benchmark

Nightly performance guard. Independent baselines — cannot modify op/tests/workloads.

- **OWNS**: profiling, baseline comparisons, [`benchmarks/`](../../benchmarks/)
- **MUST NOT WRITE**: correctness assertions, kernel code
- **MUST NOT** (oracle-leakage rule, not a write rule): import oracle/ref functions from [`tests/`](../../tests/) or [`workloads/`](../../workloads/). Benchmark-local baseline functions are allowed; the prohibition prevents the benchmark stage from sharing its correctness oracle with the test stage.

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

## Issue-authoring: declaring scope

The trust model is a semantic review lens ([`.claude/review-checklists/pre-review.md`](../../.claude/review-checklists/pre-review.md)). The pipeline's write-scope gate reads `## Constraints` bullets to learn the work's stage shape; the reviewer judges correctness against the stage contracts above. This catches same-agent fabrication of oracle + implementation while honest cross-stage work proceeds.

| Work shape                 | Constraints bullet form                                                                                         | Effect                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Joint change across stages | Behavioral / compatibility / perf bullets only                                                                  | Pipeline permits any stage; reviewer applies the lens |
| Single stage               | `Implementation-only PR.` (or `Test-only PR.`, etc.)                                                            | Pipeline confines the diff to that stage              |
| Multiple stages, declared  | One bullet per stage: `Implementation-only PR for kernel widening.` + `Test-only PR for parametrize expansion.` | Pipeline permits the named stages' union              |

Authoring rules:

- A diff-added code path with an output-distinguishing input lacking pre-existing test coverage uses the joint form so the test lands with the impl in the same PR, satisfying the reviewer's new-path-coverage criterion. Aliases (paths with no output-distinguishing input) do not force the joint form.
- Constraints is written as bulleted items — the gate parses bullets to derive declared scope.
- Pair a [`trust-model.md`](trust-model.md) citation with "separate PR" / "own PR" / "standalone PR" in one bullet to declare the named stage forbidden. Place the citation on its own bullet when no such restriction is intended.

Default when drafting from a brief: one Constraints bullet stating the behavioral or compatibility expectation. Reach for `<stage>-only PR` when the work is genuinely single-stage.

→ Template and per-section structural rules: [.foundry/mold/body-sections.md](../../.foundry/mold/body-sections.md)
