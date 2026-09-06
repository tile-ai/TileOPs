# Contributing to TileOPs

## Setup

Activate a virtual environment, then:

```bash
pip install -e '.[dev]' -c constraints.txt
pre-commit install
```

[docs/development.md](docs/development.md) covers building, testing and the dev image.

## Design first

[`docs/design/`](docs/design/) and [`src/tileops/manifest/`](src/tileops/manifest/) are the spec.
Code conforms to them; a change that does not fit the spec changes the spec first, in its own PR.
Read the document for the area before changing it, and review against the same one.

| Changing         | Read                                                                                                                                                     |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| an op or kernel  | [ops-design.md](docs/design/ops-design.md), [op-slot-rules.md](docs/design/op-slot-rules.md), [.claude/rules/code-style.md](.claude/rules/code-style.md) |
| a manifest entry | [manifest.md](docs/design/manifest.md), [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)                                   |
| a test           | [testing.md](docs/design/testing.md), [trust-model.md](docs/design/trust-model.md)                                                                       |
| a benchmark      | [testing.md § Benchmarks](docs/design/testing.md#benchmarks), [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                     |
| a design doc     | [.claude/domain-rules/design-docs.md](.claude/domain-rules/design-docs.md)                                                                               |

A rule that can be checked mechanically is a hook, not a line in a document. Run
`pre-commit run --all-files` before pushing; CI runs the same set, and what it says is the rule.

## Naming

[`.claude/conventions/types.sh`](.claude/conventions/types.sh) is the single source of truth for
types, branch names and labels — CI sources it. Conventional Commits (`feat(scope): …`) is not used.

|                   | Form                                                                |
| ----------------- | ------------------------------------------------------------------- |
| commit / PR title | `[Type] description`, or `[Type][foundry][Scope] description`       |
| branch            | `<type>/<area>/<slug>`, all lowercase — `perf/norm/rms-norm-sol`    |
| issue title       | `[TYPE][COMPONENT] short description in lowercase`, ≤ 80 characters |

## Pull requests

[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) is the body shape: what
changed and how to verify it. A PR touching `tests/` reports its test node delta, and one touching
a kernel or op reports benchmark numbers against a baseline that is not TileOPs.
