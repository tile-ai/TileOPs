# TileOPs Documentation

Design-first, spec-driven documentation for TileOPs. [`tileops/manifest/`](../tileops/manifest/) and the documents below are the authoritative spec; code conforms to the spec, not the other way around.

## Design

| Document                                                  | Scope                                                                                                                                       |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](design/architecture.md)                 | System modules (M1–M8), data flow, agent production loop, directory structure.                                                              |
| [manifest.md](design/manifest.md)                         | `tileops/manifest/` spec: signature, workloads, roofline fields, source.                                                                    |
| [ops-design.md](design/ops-design.md)                     | Op interface execution guide — how to add a new op.                                                                                         |
| [ops-design-reference.md](design/ops-design-reference.md) | Op interface detail reference: interface tables, codegen, naming, protocol.                                                                 |
| [roofline.md](design/roofline.md)                         | `tileops/manifest/` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator, benchmark, M5, op codegen). |

## Process

| Document                                | Scope                                                                                                |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [trust-model.md](design/trust-model.md) | Trust boundaries between manifest → test → implementation → benchmark; workloads layer contract.     |
| [testing.md](design/testing.md)         | Test and benchmark framework: core abstractions, tolerances, reporting rules.                        |
| [tileops-skills.md](tileops-skills.md)  | Developer decision guide: which repo-provided skill to use for which task, with composition diagram. |

## Performance Guides

- [perf/trace-timeline.md](perf/trace-timeline.md) — in-kernel timeline tracer: annotate, build, run, and read

## External

- [TileOPs.github.io](https://tile-ai.github.io/TileOPs.github.io/) — auto-generated documentation site (API reference, perf tables, support matrix).
