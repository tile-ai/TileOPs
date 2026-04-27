# CLAUDE.md

## Project Overview

TileOPs is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

This project follows **design-first, spec-driven** development: design docs and `ops_manifest.yaml` are the authoritative spec; code conforms to the spec, not the other way around.

## Development Environment

Activate a virtual environment, then `make install` (deps + pre-commit hooks).

## Key References

### Design

- [architecture.md](docs/architecture.md) — system modules (M1-M8), data flow, agent production loop, directory structure
- [ops-design.md](docs/ops-design.md) — Op interface execution guide (how to add a new op)
- [ops-design-reference.md](docs/ops-design-reference.md) — Op interface detail reference (interface tables, codegen, naming, protocol)
- [manifest.md](docs/manifest.md) — `ops_manifest.yaml` spec format (signature, workloads, roofline, source)
- [roofline.md](docs/roofline.md) — `ops_manifest.yaml` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator / benchmark / M5 / codegen)

### Process

- [trust-model.md](docs/trust-model.md) — trust boundaries (manifest → test → implementation → benchmark), workloads layer contract
- [testing.md](docs/testing.md) — test/benchmark framework, core abstractions, tolerances, reporting rules
- [tileops-skills.md](docs/tileops-skills.md) — developer decision guide: which repo-provided skill to use for which task

## Reading `ops_manifest.yaml`

This file is thousands of lines — never slurp it as text. To inspect an entry, parse it (`yaml.safe_load`) and index by op name. For edits, use a round-trip parser (`ruamel.yaml`) to preserve comments and key order. Reserve `Read`/`grep` for targeted line lookups, not structural reading.

## Domain Rules (load on demand)

Read the relevant context file **before** modifying files in that domain. Do not load them if your task does not touch that domain.

| When you modify                                                   | Read first                                                                               |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/`                                                          | [.claude/domain-rules/testing-budget.md](.claude/domain-rules/testing-budget.md)         |
| `tileops/ops_manifest.yaml`                                       | [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)           |
| `scripts/validate_manifest.py`, `tests/test_validate_manifest.py` | [.claude/domain-rules/manifest-validator.md](.claude/domain-rules/manifest-validator.md) |
| `tileops/ops/`, `tileops/kernels/`                                | [.claude/domain-rules/ops-design.md](.claude/domain-rules/ops-design.md)                 |
| `benchmarks/`                                                     | [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                   |
| `workloads/`                                                      | [docs/trust-model.md](docs/trust-model.md)                                               |
