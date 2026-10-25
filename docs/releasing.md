# Releasing

A `vX.Y.Z` tag on `main` publishes to PyPI through `.github/workflows/release.yml`. A manual run of the same workflow publishes to TestPyPI.

## Once, before the first release

- On PyPI and TestPyPI, add a trusted publisher: owner `tile-ai`, repository `TileOPs`, workflow `release.yml`, environment `pypi` (PyPI) or `testpypi` (TestPyPI).
- In the repository settings, give the `pypi` and `testpypi` environments required reviewers, and protect tags matching `v*`.

## Each release

1. Update `CHANGELOG.md`: rename `Unreleased` to the version, through a pull request.
1. Pick the `main` commit to release. It needs a successful `preflight` and `gpu-smoke` push run, and a successful `nightly` run started by hand (Actions → Nightly → Run workflow on `main`) at that commit; the release workflow refuses the tag otherwise.
1. Dry run: Actions → Release → Run workflow with version `X.Y.ZrcN`, approve the `testpypi` environment, then check the TestPyPI page and `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tileops==X.Y.ZrcN`.
1. Tag and push: `git tag vX.Y.Z <commit>` and `git push upstream vX.Y.Z`. Approve the `pypi` environment when the build and verification jobs pass.
1. Create the GitHub release from the tag, with the version's `CHANGELOG.md` section as its notes.
