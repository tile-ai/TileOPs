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

An implementation PR (the one that aligns op code with the manifest) MAY touch the manifest entry of the op being aligned, but ONLY within the metadata fields enumerated below. Contractual fields stay frozen and require a separate manifest-only PR with human review.

**Allowed in an implementation PR** (metadata only — no human re-review of the spec needed):

- Flip `status: spec-only` ↔ `status: implemented`.
- Add or remove entries in `source.kernel_map` (the dispatch registration table).

These edits are permitted ONLY on op entries whose `signature`, `workloads`, `roofline`, and `params` blocks are byte-identical between the PR's base and head.

**Not allowed in an implementation PR** (contractual — require a manifest-only PR with human review):

- Any change to `signature` (parameter names, order, default values, `dtype`, `ref_api`, `ref_dtype`).
- Any change to `workloads` (shape entries, axis names, named-tile sets).
- Any change to `roofline.*` (`vars`, formulas, `flops`, `bytes`, `peak_*`, per-consumer fields).
- Any change to `params` (declared static or dynamic parameters, defaults).
- Any change to output-dtype rules or shape rules (e.g. `output_dtype`, `shape_rules`).

This carve-out narrows the prohibition; it does not relax the trust boundary. If an implementation PR needs any change in the "not allowed" list, stop, open a manifest-only PR for that change first, and resume the implementation PR against the merged manifest.

See: [.claude/rules/manifest-trust-model.md](../../.claude/rules/manifest-trust-model.md) §Status flip carve-out (authoritative copy).

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
