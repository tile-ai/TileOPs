# Claude.md

## Project Overview

TileOPs (TOP) is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

## Development Environment

1. Clone repository: `git clone https://github.com/tile-ai/TileOPs && cd TileOPs`
1. Create and activate a virtual environment (venv, conda, etc.)
1. Install dependencies: `pip install -e '.[dev]' -v`
1. Install pre-commit hooks: `pre-commit install`

## Key References

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — architecture (4-layer stack), development workflow, coding standards, testing strategy, and PR process
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — "2+1 Review" policy, branch naming, and commit message conventions

## Collaboration Rules for Claude

- Prefer minimal, targeted changes and avoid unrelated refactoring.
- After code changes, run the most relevant tests first.
- If unrelated failures appear, report them but do not fix them in the same task.
- Add necessary docs and tests when introducing files/interfaces.
- Response should include: change summary, affected paths, validation steps, and next suggestions.

## Skill Index

| Skill                   | Trigger                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `migrating-new-op`      | Use when adding or migrating an operator to TileOPs with the required kernel->op->function->layer delivery path and validation checklist     |
| `creating-pull-request` | Use when turning local changes into a clean PR end-to-end: pre-checks (format/tests), branch/commit, PR title/labels, review and CI handling |
