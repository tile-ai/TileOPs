"""Reduction-alignment tests relocated; this stub keeps CI happy.

The substantive content of this file was deleted as part of the cleanup
landed by this PR. The direct LogSoftmax FLOP regression now lives in
``tests/ops/test_softmax.py``; the three logical-reduce behavior tests
moved verbatim to ``tests/ops/test_logical_reduce.py``.

The stub remains because CI's targeted ``PYTEST_TARGETS`` resolver
reads ``git diff --name-only`` against the base branch and includes
deleted files in the test target list. Pytest then exits 4
("file not found") and fails the gpu-smoke job. Replacing the deletion
with a module-level ``pytest.skip`` keeps the path resolvable and lets
pytest exit cleanly with one skipped item.
"""
# FIXME(staged-rollout): module-level skip stub is a CI-resolver workaround.
#
# Broken invariant: cleanup PRs cannot fully delete a test file because
# the CI targeted-scope resolver still passes the deleted path to
# pytest (exit 4). This stub is a pure no-op kept solely to satisfy
# that resolver.
# Why: fixing the resolver belongs to the CI workflow, which is out of
# scope for the test-cleanup PR (#1239 explicitly confines its diff to
# tests/ops/).
# Cleanup: once the CI's targeted-scope resolver filters paths against
# the post-merge worktree (or otherwise excludes deleted files from
# PYTEST_TARGETS), delete this file outright.
import pytest

pytest.skip(
    "tests relocated to test_softmax.py and test_logical_reduce.py",
    allow_module_level=True,
)
