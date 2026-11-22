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
| Benchmarks                                                  | workload rows, the reference in `workloads/`, the op interface (including `eval_roofline()` and `roof_key()`)              |
| Roofline tool                                               | benchmark output and the GPU profile; it never instantiates an op                                                          |
| Docs site                                                   | the manifest YAML (read without torch), op docstrings, benchmark and roofline output                                       |
| Models                                                      | the public op interface, never a private `tileops._*` module; nothing under `src/` depends on a model                      |

## Manifest

Source of truth for op interfaces: signatures (shapes, dtypes, presence,
effects), workload rows, roofline formulas, status and composition.

It carries no kernel internals, dispatch strategy, or test logic. Those are
implementation choices; freezing one into the spec makes every later
implementation conform to an accident.

→ Spec: [manifest.md](manifest.md)

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

→ Rules: [testing.md §Tests](testing.md#tests)

## Implementation

PyTorch is the spec oracle, not a runtime path. The Op layer produces results
through TileLang kernels or tensor primitives; delegating to a higher-level
PyTorch operator (`torch.sum`, `F.softmax`, …) at forward time is forbidden —
it would make the library measure and ship PyTorch. The one exception is an
operation TileLang cannot express at the required shape and no composition of
tensor primitives replaces; its call site names the blocking limitation. A
helper such as `x.float().mean(...)` is a tensor primitive, not a fallback.

→ Rules: [ops-design.md](ops-design.md), [code-style.md](../code-style.md)

## Models

[`models/`](../../models/) composes published models from the public ops. It sits
at the repository root beside `workloads/` and `benchmarks/` and does not ship in
the package: models are experimental, and their interface becomes a contract only
after validation at scale.

| Code                        | Never imports                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `src/tileops/`              | `models`, `workloads`, `tests`, `benchmarks`, or a checkpoint package (`safetensors`, `transformers`, `huggingface_hub`) |
| `models/`                   | a private `tileops._*` module                                                                                            |
| `tests/ops/`, `benchmarks/` | `models`                                                                                                                 |

- Every computation is an op call. Between calls a model uses only tensor primitives: views, copies, concatenation, index metadata and constant tables such as RoPE frequencies.
- Reading a checkpoint, parsing a model config, converting weights into the layouts the ops read, placing layers on devices and the generation loop live in `models/`.
- A runtime computation no op serves becomes a new op through the manifest, never a helper under `src/`.
- Code moves to `models/_common/` once two models carry the same code, not before.
- A package a model or its reference imports is declared in the `dev` dependency group, pinned to the runner lock's version, by the pull request that first imports it.
- A model is tested against its authors' code, never copied into the repository: [`workloads/models/`](../../workloads/models/) runs it with torch in place of any custom kernel it calls.

[`tests/test_model_imports.py`](../../tests/test_model_imports.py) checks the import table.

→ Rules: [testing.md §Model tests](testing.md#model-tests)

## Benchmark

Benchmarks consume the same workload reference and verification contract as tests,
and own timing, competitor selection and reporting. They do not import from
[`tests/`](../../tests/) or override the workload reference or numerical policy.

Each comparable implementation is verified against that reference before timing.
A torch baseline times the workload's `torch_program`, not its reference, and is
verified like any other implementation.
A competitor with different semantics must be explicitly marked incomparable;
its timing cannot produce a correctness-backed performance ratio.

Reports identify the operator that actually ran. Coverage is checked against
those runtime identities, rather than inferred from source-code patterns.

→ Rules: [testing.md §Benchmarks](testing.md#benchmarks)

## Workloads layer

[`workloads/`](../../workloads/) owns input generation, reference computation and
numerical verification. A concrete workload combines these through
`gen_inputs`, `ref_program` and `verification`; related operators may share a
parameterized workload.

`ref_program` is the correctness reference: it computes a float result at float32 or
wider precision, never below the output's, and rounds once, at its output. An
intermediate rounds earlier only where the op's contract stores or documents that
rounding. The reference does not depend on how a device's torch rounds a 16-bit step.
An optional `torch_program` is the op as torch runs it at the input dtype. It is a
performance baseline, never a reference. Manifest rows supply shapes, dtypes and other declared
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

→ Cross-refs: [testing.md](testing.md)
