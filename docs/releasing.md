# Releasing

A `vX.Y.Z` tag on `main` publishes to PyPI through `.github/workflows/release.yml`. A manual run of the same workflow publishes to TestPyPI.

## Once, before the first release

- On PyPI and TestPyPI, add a trusted publisher: owner `tile-ai`, repository `TileOPs`, workflow `release.yml`, environment `pypi` (PyPI) or `testpypi` (TestPyPI). Before the first upload the project does not exist yet, so create it as a pending publisher for the project name `tileops` under the account's Publishing page.
- In the repository settings, give the `pypi` and `testpypi` environments required reviewers, and protect tags matching `v*`.

## Each release

1. Update `CHANGELOG.md` through a pull request: move the `Unreleased` entries under `## X.Y.Z - <release date>` and leave an empty `## Unreleased` above it.
1. Pick the `main` commit to release. It needs a successful `preflight` and `gpu-smoke` push run, and a successful `nightly` run started by hand (Actions → Nightly → Run workflow on `main`) at that commit; the release workflow refuses the tag otherwise.
1. Dry run: Actions → Release → Run workflow with version `X.Y.ZrcN`, approve the `testpypi` environment, then check the TestPyPI page and `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tileops==X.Y.ZrcN`.
1. Tag and push: `git tag vX.Y.Z <commit>` and `git push upstream vX.Y.Z`. Approve the `pypi` environment when the build and verification jobs pass.
1. Create the GitHub release from the tag, with the version's `CHANGELOG.md` section as its notes.
