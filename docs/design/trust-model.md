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

Tolerances and assertions live with the test, which does not import from
[`benchmarks/`](../../benchmarks/).

Input construction does not, and neither does the reference computation of an
op that has a workload named for it: both belong in
[`workloads/`](../../workloads/). Anything left inside `tests/` is unreachable
from a benchmark — see [§Benchmark](#benchmark) — so it gets copied, and the
two copies drift.

→ Rules: [testing-budget.md](../../.claude/domain-rules/testing-budget.md) | Guide: [testing.md §Tests](testing.md#tests)

## Implementation

PyTorch is the spec oracle, not a runtime path. The Op layer produces results
through TileLang kernels or tensor primitives; delegating to a higher-level
PyTorch operator (`torch.sum`, `F.softmax`, …) at forward time is forbidden —
it would make the library measure and ship PyTorch. Narrow fallback exceptions
are listed in [ops-design.md](../../.claude/domain-rules/ops-design.md).

→ Rules: [ops-design.md](../../.claude/domain-rules/ops-design.md) | Guide: [ops-design.md](ops-design.md)

## Benchmark

A benchmark does not import from [`tests/`](../../tests/). It reads the
op's reference from [`workloads/`](../../workloads/) — the same definition the
test checks against — and times it as the torch baseline.

The `tests/` boundary buys decoupling: nightly benchmarks keep running across
test-side refactors, and a tolerance or comparator change cannot move a
baseline number. Sharing the reference costs none of that, and removes the
second copy that used to drift.

A baseline that is another idiom for the same computation overrides
`ref_program` in the benchmark and says why.

A baseline that is a different implementation is timed under its own tag next
to the reference: the tag is what names it in the report, so it is checked
against the reference before the case is timed.

Which manifest entry a benchmark measures is settled by running it, not by
reading it. The op is whatever class the benchmark constructs, so the run's
report carries that class's name and is compared against the entries declaring a
benchmark. A source check can only look for a marker, and a marker is not the op:
it goes unchecked against what ran, and it makes the shape of the source — a
literal here, a construction there — a condition of passing.

What the source does answer is the file's own contract: workloads from the
manifest, roofline from the op. That needs no op name, so no benchmark shape is
illegal.

[`benchmarks/tests/test_benchmark_boundaries.py`](../../benchmarks/tests/test_benchmark_boundaries.py)
checks the `tests/` import and a locally defined `gen_inputs`, both by literal
name.

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads layer

The shared layer, and the only one both tests and benchmarks import.

**Provides**: `WorkloadBase` (`gen_inputs`), `FixtureMeta` / `FixtureBase`
(parametrize), and one workload class per op — or one parameterized class a
family shares.

**Must contain**: input construction for every op, and — where the class is
named for an op — that op's reference computation.

**Must not contain**: tolerances, `check`, `calculate_flops` /
`calculate_memory`, or the choice of what to time against. Those are decisions,
the first three the test's and the last the benchmark's, and a decision placed
here reaches the other consumer.

The reference computation is not a decision. It is the executable form of the
manifest's `ref_api`: what the operator means, the same for whoever asks.
Assigning it to one consumer obliges the other to keep a second copy, and two
copies of the same math drift apart in silence — the benchmark then reports a
ratio against a computation no test validated.

It belongs to the narrowest shared class that names one operator. That is
normally the workload; a workload describing only an input shape — one random
tensor, a matching pair — is reused across ops, names none of them, and so its
consumers carry the reference instead. `TestBase` already declares
`ref_program` abstract, so whichever class supplies it, a test without one
cannot be instantiated.

```
WorkloadBase (workloads/workload_base.py)  # gen_inputs(), and ref_program()
  |                                        # on the classes named for an op
  ├── TestBase (tests/test_base.py)        # adds check() and tolerances
  └── concrete subclasses per op

BenchmarkBase[W] (benchmarks/)             # generic over workload type; reads
                                           # roofline off the op, not the workload
```

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
