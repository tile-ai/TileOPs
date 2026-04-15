# CLAUDE.md

## Project Overview

TileOPs is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

## Development Environment

1. Clone repository: `git clone https://github.com/tile-ai/TileOPs && cd TileOPs`
1. Create and activate a virtual environment (venv, conda, etc.)
1. Install dependencies and pre-commit hooks: `make install`

## Key References

### Design

- [architecture.md](docs/architecture.md) — system modules (M1-M8), data flow, agent production loop, directory structure
- [ops-design.md](docs/ops-design.md) — Op interface execution guide (how to add a new op)
- [ops-design-reference.md](docs/ops-design-reference.md) — Op interface detail reference (interface tables, codegen, naming, protocol)
- [manifest.md](docs/manifest.md) — `ops_manifest.yaml` spec format (signature, workloads, roofline, source)
- [roofline.md](docs/roofline.md) — performance evaluation methodology (SOL bound, efficiency ratio, GPU profiles)

### Process

- [trust-model.md](docs/trust-model.md) — trust boundaries (manifest → test → implementation → benchmark), workloads layer contract
- [testing.md](docs/testing.md) — test/benchmark framework, core abstractions, tolerances, reporting rules

### External

- [TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io) — auto-generated documentation site (API reference, perf tables, support matrix)

## Collaboration Rules for Claude

- Prefer minimal, targeted changes and avoid unrelated refactoring.
- After code changes, run the most relevant tests first.
- If unrelated failures appear, report them but do not fix them in the same task.
- Add necessary docs and tests when introducing files/interfaces.
- Response should include: change summary, affected paths, validation steps, and next suggestions.
- No GitHub issue numbers or ad-hoc issue annotations in source/test files. Issue refs belong in commit messages and PR descriptions only.
- When orchestrating sub-agents: verify clean worktree (`git diff --quiet && git diff --cached --quiet`) after each sub-agent returns. If dirty, commit on behalf using a descriptive message (e.g., `Sub-agent [name]: [task summary]`).
- Write commit messages in `[Type] description` or `[Type][Scope] description` format per `COMMIT_MSG_PATTERN` in `.claude/conventions/types.sh` (scope is optional). Not CI-enforced, but expected for readability.

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
