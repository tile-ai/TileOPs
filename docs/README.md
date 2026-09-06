# TileOPs Documentation

Design-first, spec-driven documentation for TileOPs. [`src/tileops/manifest/`](../src/tileops/manifest/) and the documents below are the authoritative spec; code conforms to the spec, not the other way around.

## Getting started

- [development.md](development.md) — install from source, the dev Docker image, test tiers, lint, benchmarks, packaging checks.

## Design

| Document                                                  | Scope                                                                                                                                           |
| --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](design/architecture.md)                 | System modules (M1–M8), data flow, agent production loop, directory structure.                                                                  |
| [manifest.md](design/manifest.md)                         | `src/tileops/manifest/` spec: signature, workloads, roofline fields, source.                                                                    |
| [ops-design.md](design/ops-design.md)                     | Op interface execution guide — how to add a new op.                                                                                             |
| [ops-design-reference.md](design/ops-design-reference.md) | Op interface detail reference: interface tables, codegen, naming, protocol.                                                                     |
| [op-slot-rules.md](design/op-slot-rules.md)               | Per-slot rules for a T2 op file (S1-S7, S12-S21): rule, example, common mistakes.                                                               |
| [roofline.md](design/roofline.md)                         | `src/tileops/manifest/` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator, benchmark, M5, op codegen). |

## Process

| Document                                | Scope                                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [trust-model.md](design/trust-model.md) | Trust boundaries between manifest → test → implementation → benchmark; workloads layer contract. |
| [testing.md](design/testing.md)         | Test and benchmark framework: core abstractions, tolerances, reporting rules.                    |

## Performance Guides

- [perf/trace-timeline.md](perf/trace-timeline.md) — in-kernel timeline tracer: annotate, build, run, and read

## External

- [TileOPs.github.io](https://tile-ai.github.io/TileOPs.github.io/) — auto-generated documentation site (API reference, perf tables, support matrix).
