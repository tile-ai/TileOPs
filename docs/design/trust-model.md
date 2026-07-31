# Trust Model

Each development stage owns a specific concern. Boundaries prevent one stage from silently weakening another's guarantees.

## Pipeline

```
Manifest → Test → Implementation → Benchmark
```

Each stage declares its trust contract using these headings:

- **OWNS** — what this stage authors. Required.
- **MUST PROVIDE** — what this stage must author *for a downstream stage*, and where it must land. Required whenever a downstream stage is forbidden from writing that content itself.
- **MUST NOT WRITE** — content this stage must not author, in any file. Required.
- **MUST NOT** — structural couplings (typically forbidden imports) that would defeat stage independence. Optional.

Every **MUST NOT WRITE** that blocks a downstream stage from content it needs MUST be paired with a **MUST PROVIDE** on the stage that owns it. A prohibition without a matching obligation does not remove the need — it converts it into a local re-implementation in the stage that was blocked, which is the coupling the prohibition was meant to prevent, minus the shared definition.

Reads are not policed; the trust model controls writes and import-level coupling, not file access. Per-stage rules live in each [domain rule file](../../.claude/domain-rules/).

## Manifest

Source of truth for op interfaces. Human-reviewed, separate PR.

- **OWNS**: op signatures, dtypes, workload shapes, roofline formulas, status, kernel_map (dispatch registration table), user-visible capability declarations (`torch_compile_fullgraph`)
- **MUST NOT WRITE**: kernel internals, dispatch strategy, or test logic

### Status flip carve-out

An implementation PR may edit only `status`, `source.kernel_map`, `source.test`, `source.bench`, (only when promoting `spec-only → implemented`) `workloads`, and (only together with its registered compile-test evidence) `torch_compile_fullgraph` on the aligned op; every other contractual field needs a separate manifest-only PR.

Full enumeration: [.claude/rules/manifest-trust-model.md](../../.claude/rules/manifest-trust-model.md) §Status flip carve-out.

→ Rules: [manifest-spec.md](../../.claude/domain-rules/manifest-spec.md) | Guide: [manifest.md](manifest.md)

## Test

PR-level correctness verification. QA writes tests against manifest spec.

- **OWNS**: ref_program, tolerances, assertions, [`tests/`](../../tests/), [`workloads/`](../../workloads/)
- **MUST PROVIDE**: every op's inputs constructible from [`workloads/<family>.py`](../../workloads/) — a class per op, or one parameterized class a family shares. Test classes compose it (`class FooTest(FooWorkload, TestBase)`); they do not define `gen_inputs` inline. The benchmark stage cannot write this layer, so an op whose workload is missing here leaves its benchmark no legal way to build inputs.
- **MUST NOT WRITE**: kernel code, benchmark logic, or performance measurements

→ Rules: [testing-budget.md](../../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Tests](testing.md#tests)

## Implementation

Kernel (L1) + Op (L2). Developer reads manifest + ref_program for behavior; high-perf optimization is independent.

- **OWNS**: TileLang kernels, op dispatch, class variable protocol
- **MUST NOT WRITE**: workload shape definitions, correctness assertions, manifest entries

PyTorch is the spec oracle, not a runtime path. The Op layer implements results through TileLang kernels or tensor primitives; delegating to higher-level PyTorch operators (`torch.sum`, `F.softmax`, ...) at forward time is forbidden. Narrow fallback exceptions live in [ops-design.md](../../.claude/domain-rules/ops-design.md).

→ Rules: [ops-design.md](../../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md)

## Benchmark

Nightly performance guard. Independent baselines — cannot modify op/tests/workloads.

- **OWNS**: profiling, baseline comparisons, [`benchmarks/`](../../benchmarks/)
- **MUST NOT WRITE**: correctness assertions, kernel code
- **MUST NOT** (import rule, not a write rule): import from [`tests/`](../../tests/), or import ref/oracle functions from [`workloads/`](../../workloads/). Buys decoupling, not cross-validation — no baseline output is compared against the test oracle. It keeps nightly benchmarks alive across test-side refactors, and keeps baseline timings stable when an oracle is edited.

Importing a workload class from [`workloads/`](../../workloads/) for input generation is the intended path, not an exception. A benchmark MUST NOT author `gen_inputs`: a missing workload is a test-stage **MUST PROVIDE** gap, closed by a `workloads/`-only PR, not by a local copy. Subclassing an imported workload to attach a benchmark-local baseline method is allowed — that method is this stage's own content.

Baselines MUST prefer an independent external implementation; a benchmark-local one is the fallback. Name it for what it is (`torch_baseline`, `flashinfer_baseline`) — `ref_program` is the test stage's correctness oracle and does not belong in this stage.

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads Layer

Shared input-definition layer — not a development stage. Test stage OWNS it (QA creates workload classes first).

**Provides**: `WorkloadBase` (gen_inputs), `FixtureMeta`/`FixtureBase` (parametrize), per-op workload subclasses.

**Must contain**: input construction for every op — the test stage's **MUST PROVIDE** obligation. One class per op or one parameterized class per family, whichever the inputs justify. Both downstream stages import from here; it is the only layer they share.

**Must not contain**: ref_program, check/tolerance logic, calculate_flops/memory, benchmark baselines. Reason: anything placed here couples the two stages that import it. Correctness logic belongs to the test stage, timing baselines to the benchmark stage.

```
WorkloadBase (workloads/workload_base.py)  # gen_inputs() only — abstract contract
  ├── TestBase (tests/test_base.py)     # adds ref_program(), check()
  └── concrete subclasses per op        # the test stage's MUST PROVIDE artifact

BenchmarkBase[W] (benchmarks/)          # generic over workload type; reads
                                        # roofline off the op, not the workload
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
