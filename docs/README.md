# TileOPs Documentation

Design-first, spec-driven documentation for TileOPs. [`src/tileops/manifest/`](../src/tileops/manifest/) and the documents below are the authoritative spec; code conforms to the spec, not the other way around.

## Using TileOPs

- [Docs site](https://tile-ai.github.io/TileOPs.github.io/) — API reference, user guides and nightly benchmarks.
- [Backends](https://tile-ai.github.io/TileOPs.github.io/backends/) — serving ops on another device or with your own kernels.

## Contributing

- [development.md](development.md) — install from source, the dev Docker image, test tiers, lint, benchmarks, packaging checks.
- [code-style.md](code-style.md) — code rules a reader applies by hand; the mechanical ones are `pre-commit` hooks.
- [stability.md](stability.md) — public modules, compatibility promise, support scope, writing an op outside TileOPs.

## Design

| Document                                                  | Scope                                                                                                                                                      |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [manifest.md](design/manifest.md)                         | `src/tileops/manifest/` spec: signature, workloads, roofline fields, composition.                                                                          |
| [ops-design.md](design/ops-design.md)                     | Op interface execution guide — how to add a new op.                                                                                                        |
| [ops-design-reference.md](design/ops-design-reference.md) | Op interface detail reference: interface tables, codegen, naming, protocol.                                                                                |
| [op-slot-rules.md](design/op-slot-rules.md)               | Per-slot rules for a T2 op file (S1-S7, S12-S20): rule, example, common mistakes.                                                                          |
| [roofline.md](design/roofline.md)                         | `src/tileops/manifest/` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator, benchmark, roofline tool, op codegen). |

## Process

| Document                                          | Scope                                                                                                                            |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [layer-boundaries.md](design/layer-boundaries.md) | What the manifest, op library, tests and benchmarks each own, and the interfaces they compose through; workloads layer contract. |
| [testing.md](design/testing.md)                   | Test and benchmark framework: core abstractions, tolerances, reporting rules.                                                    |

## Performance Guides

- [perf/trace-timeline.md](perf/trace-timeline.md) — in-kernel timeline tracer: annotate, build, run, and read

## External

- [TileOPs.github.io](https://tile-ai.github.io/TileOPs.github.io/) — the documentation site: API reference, user guides and nightly benchmarks.
