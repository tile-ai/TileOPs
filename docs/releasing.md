# Releasing

A `vX.Y.Z` tag on `main` publishes to PyPI through `.github/workflows/release.yml`. A manual run of the same workflow publishes a release candidate to TestPyPI.

1. Through a pull request, move the `Unreleased` entries of `CHANGELOG.md` under `## X.Y.Z - <release date>`.
1. Pick a `main` commit with successful `preflight`, `gpu-smoke` and manually started `nightly` runs; the release workflow refuses a tag on any other.
1. Publish `X.Y.ZrcN` to TestPyPI from that commit and install it from there.
1. Push the tag `vX.Y.Z` on that commit.
1. Create the GitHub release from the tag, with the version's `CHANGELOG.md` section as its notes.
