# CLAUDE.md

## Project Overview

TileOPs is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

This project follows **design-first, spec-driven** development: design docs and `src/tileops/manifest/` are the authoritative spec; code conforms to the spec, not the other way around.

## Development Environment

Activate a virtual environment, then `pip install -e '.[dev]' -c constraints.txt && pre-commit install`. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/development.md](docs/development.md).

## Key References

### Design

- [architecture.md](docs/design/architecture.md) — system modules (M1-M8), data flow, agent production loop, directory structure
- [ops-design.md](docs/design/ops-design.md) — Op interface execution guide (how to add a new op)
- [ops-design-reference.md](docs/design/ops-design-reference.md) — Op interface detail reference (interface tables, codegen, naming, protocol)
- [op-slot-rules.md](docs/design/op-slot-rules.md) — per-slot rules for a T2 op file (S1-S7, S12-S21), with examples and common mistakes
- [manifest.md](docs/design/manifest.md) — `src/tileops/manifest/` spec format (signature, workloads, roofline, source)
- [roofline.md](docs/design/roofline.md) — `src/tileops/manifest/` `roofline` field spec: performance model, authoring, and per-consumer contracts (validator / benchmark / M5 / codegen)

### Process

- [trust-model.md](docs/design/trust-model.md) — trust boundaries (manifest → test → implementation → benchmark), workloads layer contract
- [testing.md](docs/design/testing.md) — test/benchmark framework, core abstractions, tolerances, reporting rules

## Reading the ops manifest

The manifest lives at `src/tileops/manifest/`, one or more YAML files per op family — most families use a single file; large families may be sharded across multiple files. The `tileops.manifest` package merges them into a single `ops` dict at runtime.

- **Programmatic reads**: prefer `from tileops.manifest import load_manifest, load_workloads`. Never re-implement the merge.
- **Structural inspection**: parse the relevant family file with `yaml.safe_load` and index `ops` by op name. Pick the file from the op's family field rather than scanning all of them.
- **Edits**: edit the single family file that owns the op. Use a round-trip parser (`ruamel.yaml`) to preserve comments and key order. Op names must remain unique across files — duplicates raise at load time.
- Reserve `Read`/`grep` for targeted line lookups inside one family file, not structural reading.

## Domain Rules (load on demand)

Read the relevant context file **before** modifying files in that domain. Do not load them if your task does not touch that domain.

| When you modify                                                   | Read first                                                                               |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `tests/`                                                          | [.claude/domain-rules/testing-budget.md](.claude/domain-rules/testing-budget.md)         |
| `src/tileops/manifest/`                                           | [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)           |
| `scripts/validate_manifest.py`, `tests/test_validate_manifest.py` | [.claude/domain-rules/manifest-validator.md](.claude/domain-rules/manifest-validator.md) |
| `src/tileops/ops/`, `src/tileops/kernels/`                        | [.claude/domain-rules/ops-design.md](.claude/domain-rules/ops-design.md)                 |
| `benchmarks/`                                                     | [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                   |
| `workloads/`                                                      | [docs/design/trust-model.md](docs/design/trust-model.md)                                 |
| `docs/design/`                                                    | [.claude/domain-rules/design-docs.md](.claude/domain-rules/design-docs.md)               |

## Contributing

Naming, PR body shape, and what a review checks: [CONTRIBUTING.md](CONTRIBUTING.md). A rule that
can be checked mechanically is a `pre-commit` hook, not a line in a document — run
`pre-commit run --all-files` and read what it says rather than re-deriving the rule.
