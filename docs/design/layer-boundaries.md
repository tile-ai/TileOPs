# Layer Boundaries

Each layer depends only on the manifest, the op interface,
[`workloads/`](../../workloads/) and other layers' published outputs, never on
their internals, so each can be replaced without touching the others.
[§Dependencies](#dependencies) lists what each one depends on; the sections after
it state what one layer owns and what it must not reach into.

## Dependencies

| Layer                                                       | Depends on                                                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Validator (CI)                                              | the manifest and the roofline analysis; for an implemented entry, the op's public interface and its `roofline.func` module |
| Generated checks, fake, operator schemas, `eval_roofline()` | the signature; `roofline` for `eval_roofline()`                                                                            |
| Implementation                                              | the signature, through the generated checks that wrap `forward`                                                            |
| Tests                                                       | workload rows, the reference in `workloads/`, the op interface                                                             |
| Benchmarks                                                  | workload rows, the reference in `workloads/`, the op interface (including `eval_roofline()` and `compute_roof()`)          |
| Roofline tool (M5)                                          | benchmark output and the GPU profile; it never instantiates an op                                                          |
| Docs site                                                   | the manifest YAML (read without torch), op docstrings, benchmark and roofline output                                       |

## Manifest

Source of truth for op interfaces: signatures (shapes, dtypes, presence,
effects), workload rows, roofline formulas, status and composition.

It carries no kernel internals, dispatch strategy, or test logic. Those are
implementation choices; freezing one into the spec makes every later
implementation conform to an accident.

→ Rules: [manifest-spec.md](../../.claude/domain-rules/manifest-spec.md) | Guide: [manifest.md](manifest.md)

## Test

Tests own fixtures, pytest outcomes and assertions about behavior, such as aliasing,
cache reuse and rejection. Numerical op-vs-reference checks consume the workload's
`ref_program` and `verification` through `check()` or `compare_outputs()`; tests do
not define a second tolerance or comparator. Tests do not import from
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
test-side refactors. Correctness uses the same workload declaration in both
consumers; timing remains the benchmark's responsibility. A policy change
therefore changes validation everywhere, at one explicit location.

A baseline that is another idiom for the same computation, or a different
implementation, is timed under its own tag next to the reference: the tag is
what names it in the report, so it is checked against the reference before the
case is timed. A benchmark never overrides `ref_program`, because the row timing
the reference would then time something no test validated.

Which manifest entry a benchmark measures is settled by running it, not by
reading it. The op is whatever class the benchmark constructs, so the run's
report carries that class's name and is compared against the ops the bench-file→op
mapping assigns to that file. A source check can only look for a marker, and a marker is not the op:
it goes unchecked against what ran, and it makes the shape of the source — a
literal here, a construction there — a condition of passing.

What the source does answer is the file's own contract: workloads from the
manifest, roofline from the op. That needs no op name, so no benchmark shape is
illegal.

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads layer

The shared layer, and the only one both tests and benchmarks import.

**Provides**: `WorkloadBase` (`gen_inputs`), `FixtureMeta` / `FixtureBase`
(parametrize), and one workload class per op — or one parameterized class a
family shares.

**Must contain**: the reference computation of the op a class is named for,
and input construction the entry's workload rows do not determine. Shapes,
dtypes, presence and metadata values come from instantiating the rows
([manifest.md § Workloads](manifest.md#workloads)).

**Must contain**: `verification(*inputs)`, when the default exact comparison with
per-output dtype tolerances is insufficient. The declaration and `ref_program`
belong to the narrowest shared class naming an operator. Shape-only bases are
extended here, never by a consumer-local reference or comparator.

**Must not contain**: pytest outcomes, `check`, timing, roofline calculations,
or the choice of benchmark competitors.

The only execution and comparison implementation is `workloads/numerics.py`:

- `verify(subject, inputs, *, reference, evidence, subject_inputs=None)` executes
  the reference and subject, restores inputs on success and failure, and runs
  declared negative controls.
- `compare_outputs(produced, expected, evidence)` checks output structure and
  numerical policy. `Exact` covers every output; `Partial` names an unchecked
  suffix; `Custom` supplies a numerical assertion after mandatory structure checks.
  A statistical `Custom` may additionally declare a `probe(subject, inputs)` for
  repeated draws. Unavailable verification must be explicitly declared.
- `CheckResult` carries checked/total output counts, the diagnostic maximum error,
  and an unchecked reason. A metric alone is never proof of comparison.

`TestBase.check(op, *inputs, runs=None)` adapts this result to pytest and JUnit.
`OpBenchmark.compare(functors, *inputs, ...)` reads the same declaration once for
all tags, then owns timing and reporting. Neither accepts numerical overrides.
An external competitor with different semantics may be explicitly named in
`noncomparable={tag: reason}`; its timing has no correctness-backed ratio.

```text
Concrete workload: gen_inputs + ref_program + verification
             |                      |
       TestBase.check       OpBenchmark.compare
             |                      |
             +------ verify --------+
                       |
                compare_outputs
                       |
                  CheckResult
```

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
