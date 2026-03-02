# Claude.md

## Project Overview

TileOPs (TOP) is a high-performance LLM operator library built on TileLang. The goal is to provide efficient, modular, and maintainable AI workload implementations.

## Development Environment

1. Clone repository: `git clone https://github.com/tile-ai/TileOPs && cd TileOPs`
1. Create and activate a virtual environment (venv, conda, etc.)
1. Install dependencies: `pip install -e '.[dev]' -v`
1. Install pre-commit hooks: `pre-commit install`

## Key References

- [DEVELOPMENT.md](docs/DEVELOPMENT.md) — architecture (2-layer stack), development workflow, coding standards, testing strategy, and PR process
- [CONTRIBUTING.md](docs/CONTRIBUTING.md) — "2+1 Review" policy, branch naming, and commit message conventions

## Collaboration Rules for Claude

- Prefer minimal, targeted changes and avoid unrelated refactoring.
- After code changes, run the most relevant tests first.
- If unrelated failures appear, report them but do not fix them in the same task.
- Add necessary docs and tests when introducing files/interfaces.
- Response should include: change summary, affected paths, validation steps, and next suggestions.

## Skill Index

| Skill                    | Trigger                                                                                                                 |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| `migrating-new-op`       | Use when adding or migrating an operator to TileOPs with the required kernel->op delivery path and validation checklist |
| `committing-changes`     | Use when creating a clean commit: pre-commit validation, branch naming, commit message format, push                     |
| `creating-pull-request`  | Use when creating a PR from a pushed branch: title/body format, labels, metadata validation                             |
| `lifecycle-pull-request` | Full PR lifecycle end-to-end: commit, create PR, monitor CI, handle reviews (composes the above two skills)             |
| `creating-issue`         | Use when filing a GitHub issue: title format, body structure, and TileOPs conventions                                   |
| `lifecycle-issue-fixer`  | End-to-end issue resolution: read issue, worktree, explore, TDD, verify, and create PR with full lifecycle              |
