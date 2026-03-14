"""Verify that no assert statements are used for runtime input validation
in tileops/kernels and tileops/ops, and that no hardcoded issue references
exist in source/test/benchmark files.

These tests encode the compliance requirements from the op-compliance-checklist.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Files that are explicitly scoped for assert replacement
KERNEL_DIRS = [
    ROOT / "tileops" / "kernels",
    ROOT / "tileops" / "ops",
]

# Directories to check for issue references
ISSUE_REF_DIRS = [
    ROOT / "tileops",
    ROOT / "tests",
    ROOT / "benchmarks",
]


def _collect_py_files(dirs):
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(set(files))


class TestNoAssertValidation:
    """No assert statements for runtime input validation in kernels/ops."""

    @pytest.mark.smoke
    def test_no_validation_asserts_in_kernels_and_ops(self):
        """Every assert with a message in tileops/kernels and tileops/ops
        should have been replaced with if/raise ValueError."""
        violations = []
        for fpath in _collect_py_files(KERNEL_DIRS):
            rel = fpath.relative_to(ROOT)
            rel_str = str(rel)
            # utils.py asserts are internal invariants, leave untouched
            if rel_str.endswith("utils.py"):
                continue
            # fp8_lighting_indexer.py assert is a post-condition check, not in scope
            if "fp8_lighting_indexer" in rel_str:
                continue

            source = fpath.read_text()
            try:
                tree = ast.parse(source, filename=str(fpath))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Assert) and node.msg is not None:
                    violations.append(
                        f"{rel}:{node.lineno}: "
                        f"assert with message (should be if/raise ValueError)"
                    )

        assert not violations, (
            "Found assert statements used for input validation "
            "(should be if/raise ValueError):\n" + "\n".join(violations)
        )


class TestNoIssueReferences:
    """No hardcoded issue #NNN references in source/test/benchmark files."""

    @pytest.mark.smoke
    def test_no_issue_refs(self):
        """No 'issue #NNN' patterns in .py files under tileops/, tests/, benchmarks/."""
        pattern = re.compile(r"issue\s*#\d+", re.IGNORECASE)
        violations = []
        for fpath in _collect_py_files(ISSUE_REF_DIRS):
            # Skip this test file itself
            if fpath.resolve() == Path(__file__).resolve():
                continue
            source = fpath.read_text()
            for i, line in enumerate(source.splitlines(), 1):
                if pattern.search(line):
                    violations.append(f"{fpath.relative_to(ROOT)}:{i}: {line.strip()}")

        assert not violations, (
            "Found hardcoded issue references:\n" + "\n".join(violations)
        )
