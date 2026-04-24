# TileOPs Documentation

Design-first, spec-driven documentation for TileOPs. [`ops_manifest.yaml`](../tileops/ops_manifest.yaml) and the documents below are the authoritative spec; code conforms to the spec, not the other way around.

## Design

| Document                                           | Scope                                                                                                                                       |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](architecture.md)                 | System modules (M1–M8), data flow, agent production loop, directory structure.                                                              |
| [manifest.md](manifest.md)                         | `ops_manifest.yaml` spec: signature, workloads, roofline fields, source.                                                                    |
| [ops-design.md](ops-design.md)                     | Op interface execution guide — how to add a new op.                                                                                         |
| [ops-design-reference.md](ops-design-reference.md) | Op interface detail reference: interface tables, codegen, naming, protocol.                                                                 |
| [roofline.md](roofline.md)                         | `ops_manifest.yaml` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator, benchmark, M5, op codegen). |

## Process

| Document                         | Scope                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------ |
| [trust-model.md](trust-model.md) | Trust boundaries between manifest → test → implementation → benchmark; workloads layer contract. |
| [testing.md](testing.md)         | Test and benchmark framework: core abstractions, tolerances, reporting rules.                    |

## Performance Guides

Empirical performance guidance lives under [`perf/`](perf/README.md). Each op category has a checklist plus the evidence backing it.

- [perf/elementwise.md](perf/elementwise.md) — elementwise kernel performance checklist
- [perf/elementwise-evidence.md](perf/elementwise-evidence.md) — reasoning and measurements

## External

- [TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io) — auto-generated documentation site (API reference, perf tables, support matrix).
