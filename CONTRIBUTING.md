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

A version tag on `main` publishes to PyPI; a `workflow_dispatch` of
[release.yml](.github/workflows/release.yml) publishes the version it is given to TestPyPI.
Both run `build` → `verify` → `publish`, and `publish` uploads the files `verify` tested
from an environment whose reviewer approves the upload.

Neither environment exists yet. Before the first dispatch or tag, a repository owner
confirms that the account holds the existing `tileops` project on pypi.org, creates the
`pypi` and `testpypi` environments with required reviewers and no self-approval, restricts
`pypi` to release tags, and registers the matching trusted publisher on pypi.org and on
test.pypi.org (owner `tile-ai`, repository `TileOPs`, workflow `release.yml`, environment
`pypi` / `testpypi`).

1. Dispatch nightly on `main` and release the commit it ran on, once it is green. The tag
   job refuses a commit without a green nightly dispatch on `main`, and a commit `main` has
   moved past can no longer get one; release a newer commit instead.
1. Dispatch `release.yml` with an `rc` version, which publishes to TestPyPI. Check it in a
   clean environment where TestPyPI serves nothing but tileops: install the dependencies
   `pyproject.toml` declares from PyPI first, then
   `pip install --no-deps --index-url https://test.pypi.org/simple/ tileops==<X.Y.Z>rc<N>`.
   Run `pip check`, and check the version, the imports and a short smoke run.
1. Draft the GitHub release notes. The generated draft groups merged PRs by the type label
   [release.yml](.github/release.yml) maps; write the title, a `pip install` line, the
   highlights, the GPU architectures and software versions this release was tested on, and
   the thanks to outside contributors by hand.
1. Push the tag (`v<X.Y.Z>` or `v<X.Y.Z>rc<N>`) and approve the `pypi` environment.
1. Install `tileops==<X.Y.Z>` from PyPI into a new environment. Check the version, the
   SHA-256 of the downloaded files against the release run's job summary, `pip check`, the
   imports, the shipped resources and a short smoke run.
1. Publish the release notes.

A published version cannot be replaced: PyPI refuses a second upload of one version. A
release found broken is yanked, and the fix goes out under a new version number.
