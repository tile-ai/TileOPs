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
Code conforms to them; a change that does not fit the spec changes the spec with it.
Read the document for the area before changing it, and review against the same one.

| Changing         | Read                                                                                                                                                     |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| an op or kernel  | [ops-design.md](docs/design/ops-design.md), [op-slot-rules.md](docs/design/op-slot-rules.md), [.claude/rules/code-style.md](.claude/rules/code-style.md) |
| a manifest entry | [manifest.md](docs/design/manifest.md), [.claude/domain-rules/manifest-spec.md](.claude/domain-rules/manifest-spec.md)                                   |
| a test           | [testing.md](docs/design/testing.md), [layer-boundaries.md](docs/design/layer-boundaries.md)                                                             |
| a benchmark      | [testing.md § Benchmarks](docs/design/testing.md#benchmarks), [.claude/domain-rules/benchmark.md](.claude/domain-rules/benchmark.md)                     |
| a design doc     | [.claude/domain-rules/design-docs.md](.claude/domain-rules/design-docs.md)                                                                               |

A rule that can be checked mechanically is a hook, not a line in a document. Run
`pre-commit run --all-files` before pushing; CI runs the same set, and what it says is the rule.
A justified exception carries `# lint: <rule> allow: <reason>`, with the rule its finding names,
on the line the finding points at (a constant's or a manifest row's first line); it exempts that
line from that hook only, and a marker without a reason exempts nothing. A file the `file-set`
hook rejects carries `lint: file-set allow: <reason>` in a comment, outside any Markdown code
fence.

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

## Releasing

A `workflow_dispatch` of [release.yml](.github/workflows/release.yml) publishes the given
version to TestPyPI; a `v<X.Y.Z>` or `v<X.Y.Z>rc<N>` tag on `main` publishes to PyPI. Both
test the built wheel on an SM90 runner and upload once a reviewer approves the environment.
`pypi` needs a reviewer other than the person who pushed the tag.

1. Dispatch nightly on `main`, and tag only a commit it ran green on.
1. Dispatch `release.yml` with an `rc` version. In a clean environment, install the declared
   dependencies from PyPI, then
   `pip install --no-deps --index-url https://test.pypi.org/simple/ tileops==<version>`.
1. Push the tag and have another reviewer approve `pypi`.
1. Install the release from PyPI in a clean environment, then publish the GitHub release notes.

PyPI refuses a second upload of a version: a broken release is yanked and fixed under a new one.
