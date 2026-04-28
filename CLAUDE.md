# CLAUDE.md

## Project Overview

TileOPs is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

This project follows **design-first, spec-driven** development: design docs and `tileops/manifest/` are the authoritative spec; code conforms to the spec, not the other way around.

## Development Environment

Activate a virtual environment, then `make install` (deps + pre-commit hooks).

## Key References

### Design

- [architecture.md](docs/architecture.md) — system modules (M1-M8), data flow, agent production loop, directory structure
- [ops-design.md](docs/ops-design.md) — Op interface execution guide (how to add a new op)
- [ops-design-reference.md](docs/ops-design-reference.md) — Op interface detail reference (interface tables, codegen, naming, protocol)
- [manifest.md](docs/manifest.md) — `tileops/manifest/` spec format (signature, workloads, roofline, source)
- [roofline.md](docs/roofline.md) — `tileops/manifest/` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator / benchmark / M5 / codegen)

### Process

- [trust-model.md](docs/trust-model.md) — trust boundaries (manifest → test → implementation → benchmark), workloads layer contract
- [testing.md](docs/testing.md) — test/benchmark framework, core abstractions, tolerances, reporting rules
- [tileops-skills.md](docs/tileops-skills.md) — developer decision guide: which repo-provided skill to use for which task

## Reading the ops manifest

The manifest lives at `tileops/manifest/`, one or more YAML files per op family (e.g. `reduction.yaml`; the `elementwise` family is sharded across `elementwise_unary_math.yaml`, `elementwise_unary_activation.yaml`, `elementwise_binary.yaml`, `elementwise_multi_input.yaml`). The `tileops.manifest` package merges them into a single `ops` dict at runtime.

- **Programmatic reads**: prefer `from tileops.manifest import load_manifest, load_workloads`. Never re-implement the merge.
- **Structural inspection**: parse the relevant family file with `yaml.safe_load` and index `ops` by op name. Pick the file from the op's family field rather than scanning all of them.
- **Edits**: edit the single family file that owns the op. Use a round-trip parser (`ruamel.yaml`) to preserve comments and key order. Op names must remain unique across files — duplicates raise at load time.
- Reserve `Read`/`grep` for targeted line lookups inside one family file, not structural reading.

## Domain Rules (load on demand)

Read the relevant context file **before** modifying files in that domain. Do not load them if your task does not touch that domain.

| When you modify                                                   | Read first                                                                               |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/`                                                          | [.claude/domain-rules/testing-budget.md](.claude/domain-rules/testing-budget.md)         |
| `tileops/manifest/`                                               | [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)           |
| `scripts/validate_manifest.py`, `tests/test_validate_manifest.py` | [.claude/domain-rules/manifest-validator.md](.claude/domain-rules/manifest-validator.md) |
| `tileops/ops/`, `tileops/kernels/`                                | [.claude/domain-rules/ops-design.md](.claude/domain-rules/ops-design.md)                 |
| `benchmarks/`                                                     | [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                   |
| `workloads/`                                                      | [docs/trust-model.md](docs/trust-model.md)                                               |
