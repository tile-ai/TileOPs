# Claude.md

## Project Overview

TileOPs (TOP) is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

## Development Environment

1. Clone repository: `git clone https://github.com/tile-ai/TileOPs && cd TileOPs`
1. Activate environment: `conda activate top`
1. Install dependencies: `pip install -e '.[dev]' -v`
1. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a branch from `main`:
   - Feature: `feature/<name>`
   - Fix: `fix/<description>`
   - Release: `release/<version>`
1. Follow coding conventions:
   - 4-space indentation
   - 100-character line width
   - Type hints required
   - Google-style docstrings
1. Add/update tests for changes and keep runs reproducible (fixed random seed)
1. Validate locally:
   - Single test example: `PYTHONPATH="$PWD" pytest tests/ops/test_xxx.py`
   - Full test suite: `PYTHONPATH="$PWD" pytest tests/`
1. Run checks before commit: `pre-commit run --all-files`

## Architecture Organization

When adding new code, follow this stack and direction strictly:

`kernel -> op -> function -> layer`

- `kernel`: low-level compute kernels (TileLang/Triton/CUDA-like implementation details)
- `op`: operator wrappers and execution contracts around kernels
- `function`: user-facing functional APIs that compose one or more ops
- `layer`: nn.Module-style abstractions for model integration

Rules:

- Build from bottom to top: implement `kernel` first, then expose via `op`, then wire into `function`, and finally add `layer`.
- Keep dependency direction one-way (upper layers can depend on lower layers; lower layers must not import upper layers).
- Place tests at the corresponding level (`tests/ops`, `tests/functions`, `tests/layers`) and add kernel-related checks where applicable.
- Do not bypass intermediate layers unless there is an explicit performance or API reason documented in the PR.

## Collaboration Guide

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for the development workflow, architecture, and coding standards. See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for the "2+1 Review" policy, branch naming, and commit message conventions.

## Collaboration Rules for Claude

- Prefer minimal, targeted changes and avoid unrelated refactoring.
- After code changes, run the most relevant tests first.
- If unrelated failures appear, report them but do not fix them in the same task.
- Add necessary docs and tests when introducing files/interfaces.
- Response should include: change summary, affected paths, validation steps, and next suggestions.

## Skill Index

| Skill              | Trigger                                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `migrating-new-op` | Use when adding or migrating an operator to TileOPs with the required kernel->op->function->layer delivery path and validation checklist |
