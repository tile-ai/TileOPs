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

Benchmarks consume the same workload reference and verification contract as tests,
and own timing, competitor selection and reporting. They do not import from
[`tests/`](../../tests/) or override the workload reference or numerical policy.

Each comparable implementation is verified against that reference before timing.
A competitor with different semantics must be explicitly marked incomparable;
its timing cannot produce a correctness-backed performance ratio.

Reports identify the operator that actually ran. Coverage is checked against
those runtime identities, rather than inferred from source-code patterns.

→ Rules: [benchmark.md](../../.claude/domain-rules/benchmark.md) | Guide: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads layer

[`workloads/`](../../workloads/) owns input generation, reference computation and
numerical verification. A concrete workload combines these through
`gen_inputs`, `ref_program` and `verification`; related operators may share a
parameterized workload. Manifest rows supply shapes, dtypes and other declared
parameters. Numerical policy belongs at the narrowest shared workload boundary,
so changing it changes validation for both tests and benchmarks.

Declared state updates are part of the observable result, alongside returned
tensors. Workloads expose the same result structure for the implementation and
reference. Partial or unavailable verification must be explicit; a timing or an
error metric alone does not establish correctness.

[`numerics.py`](../../workloads/numerics.py) owns execution and comparison.
`verify` handles input isolation, restoration and declared controls or repeated
sampling; `compare_outputs` checks already-produced results against the same
contract. Both consumers use this shared machinery:

```text
Workload: inputs + reference + verification
                    |
       TestBase.check / bench.Runner.compare
                    |
                  verify
                    |
              compare_outputs
```

Workloads do not own pytest outcomes, timing, roofline calculations or benchmark
competitor selection. Those remain at the consumer boundary.

→ Cross-refs: [architecture.md](architecture.md), [testing.md](testing.md)
