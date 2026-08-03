# Layering

Where each kind of content lives, and the couplings that would defeat that.
Each boundary below exists because crossing it produced duplication, drift, or
a false guarantee.

## Manifest

Source of truth for op interfaces: signatures, dtypes, workload shapes,
roofline formulas, status, the `kernel_map` dispatch registration table, and
user-visible capability declarations (`torch_compile_fullgraph`).

It carries no kernel internals, dispatch strategy, or test logic. Those are
implementation choices; freezing one into the spec makes every later
implementation conform to an accident.

→ Rules: [manifest-spec.md](../../.claude/domain-rules/manifest-spec.md) | Guide: [manifest.md](manifest.md)

## Test

`ref_program`, tolerances and assertions live with the test.

Input construction does not: it belongs in [`workloads/`](../../workloads/). A
workload left inside `tests/` is unreachable from a benchmark — see
[§Benchmark](#benchmark) — so it gets copied, and the two copies drift.

→ Rules: [testing-budget.md](../../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Tests](testing.md#tests)

## Implementation

PyTorch is the spec oracle, not a runtime path. The Op layer produces results
through TileLang kernels or tensor primitives; delegating to a higher-level
PyTorch operator (`torch.sum`, `F.softmax`, …) at forward time is forbidden —
it would make the library measure and ship PyTorch. Narrow fallback exceptions
are listed in [ops-design.md](../../.claude/domain-rules/ops-design.md).

→ Rules: [ops-design.md](../../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md)

## Benchmark

A benchmark does not import from [`tests/`](../../tests/), and does not import
ref or oracle functions from [`workloads/`](../../workloads/).

This buys decoupling, not cross-validation — no baseline output is compared
against the test oracle. It keeps nightly benchmarks running across test-side
refactors, and keeps baseline timings stable when an oracle is edited.

Enforced by [`benchmarks/tests/test_benchmark_boundaries.py`](../../benchmarks/tests/test_benchmark_boundaries.py).

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads layer

The shared input-definition layer, and the only one both tests and benchmarks
import.

**Provides**: `WorkloadBase` (`gen_inputs`), `FixtureMeta` / `FixtureBase`
(parametrize), and one workload class per op — or one parameterized class a
family shares.

**Must contain**: input construction for every op.

**Must not contain**: `ref_program`, check or tolerance logic,
`calculate_flops` / `calculate_memory`, benchmark baselines. Anything placed
here couples the two sides that import it: correctness logic belongs with the
test, timing baselines with the benchmark.

```
WorkloadBase (workloads/workload_base.py)  # gen_inputs() only — abstract contract
  ├── TestBase (tests/test_base.py)        # adds ref_program(), check()
  └── concrete subclasses per op

BenchmarkBase[W] (benchmarks/)             # generic over workload type; reads
                                           # roofline off the op, not the workload
```

Enforced by [`tests/test_workload_placement.py`](../../tests/test_workload_placement.py).

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
