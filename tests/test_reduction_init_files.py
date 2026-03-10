"""Tests for reduction placeholder __init__.py files (issue #432).

Validates that:
- tileops/kernels/reduction/__init__.py has commented-out imports for 6 Kernel classes
- tileops/ops/reduction/__init__.py has commented-out imports for 20 Op classes
- tileops/ops/__init__.py has commented-out reduction imports and __all__ entries
- All files pass ruff check and ruff format --check
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parent.parent

KERNEL_INIT = ROOT / "tileops" / "kernels" / "reduction" / "__init__.py"
OPS_REDUCTION_INIT = ROOT / "tileops" / "ops" / "reduction" / "__init__.py"
OPS_INIT = ROOT / "tileops" / "ops" / "__init__.py"

# --- Expected kernel classes (6) ---

KERNEL_CLASSES = [
    "ArgreduceKernel",
    "CumulativeKernel",
    "LogicalReduceKernel",
    "ReduceKernel",
    "SoftmaxKernel",
    "VectorNormKernel",
]

# --- Expected op classes (20) ---

OP_CLASSES = [
    "AllOp",
    "AnyOp",
    "ArgmaxOp",
    "ArgminOp",
    "CountNonzeroOp",
    "CummaxOp",
    "CumminOp",
    "CumprodOp",
    "CumsumOp",
    "InfNormOp",
    "L1NormOp",
    "L2NormOp",
    "LogSoftmaxOp",
    "LogSumExpOp",
    "ReduceMaxOp",
    "ReduceMeanOp",
    "ReduceMinOp",
    "ReduceProdOp",
    "ReduceSumOp",
    "SoftmaxOp",
]


class TestKernelReductionInit:
    """Tests for tileops/kernels/reduction/__init__.py."""

    def test_file_exists(self):
        assert KERNEL_INIT.exists(), f"{KERNEL_INIT} does not exist"

    def test_has_all_kernel_classes_commented(self):
        content = KERNEL_INIT.read_text()
        for cls in KERNEL_CLASSES:
            assert cls in content, f"Kernel class {cls!r} not found in {KERNEL_INIT}"

    def test_imports_are_commented_out(self):
        content = KERNEL_INIT.read_text()
        for cls in KERNEL_CLASSES:
            # The class should appear in a commented-out import line
            found_commented = False
            for line in content.splitlines():
                if cls in line and line.lstrip().startswith("#"):
                    found_commented = True
                    break
            assert found_commented, f"Kernel class {cls!r} should be in a commented-out import"

    def test_has_all_dunder_all(self):
        content = KERNEL_INIT.read_text()
        assert "__all__" in content, "__all__ not found"

    def test_all_entries_are_commented_out(self):
        content = KERNEL_INIT.read_text()
        for cls in KERNEL_CLASSES:
            # Each class should appear in __all__ as a commented-out string entry
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Kernel class {cls!r} should have a commented-out __all__ entry"
            )

    def test_exact_kernel_count(self):
        content = KERNEL_INIT.read_text()
        # Count commented-out __all__ entries
        commented_all = [
            line
            for line in content.splitlines()
            if line.lstrip().startswith("#") and '"' in line and "Kernel" in line
        ]
        # Should be at least 6 kernel classes in __all__
        assert len(commented_all) >= len(KERNEL_CLASSES), (
            f"Expected at least {len(KERNEL_CLASSES)} commented kernel entries, "
            f"found {len(commented_all)}"
        )


class TestOpsReductionInit:
    """Tests for tileops/ops/reduction/__init__.py."""

    def test_file_exists(self):
        assert OPS_REDUCTION_INIT.exists(), f"{OPS_REDUCTION_INIT} does not exist"

    def test_has_all_op_classes_commented(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in OP_CLASSES:
            assert cls in content, f"Op class {cls!r} not found in {OPS_REDUCTION_INIT}"

    def test_imports_are_commented_out(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in OP_CLASSES:
            found_commented = False
            for line in content.splitlines():
                if cls in line and line.lstrip().startswith("#"):
                    found_commented = True
                    break
            assert found_commented, f"Op class {cls!r} should be in a commented-out import"

    def test_has_all_dunder_all(self):
        content = OPS_REDUCTION_INIT.read_text()
        assert "__all__" in content, "__all__ not found"

    def test_all_entries_are_commented_out(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in OP_CLASSES:
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Op class {cls!r} should have a commented-out __all__ entry"
            )

    def test_exact_op_count(self):
        content = OPS_REDUCTION_INIT.read_text()
        commented_all = [
            line
            for line in content.splitlines()
            if line.lstrip().startswith("#") and '"' in line and "Op" in line
        ]
        assert len(commented_all) >= len(OP_CLASSES), (
            f"Expected at least {len(OP_CLASSES)} commented op entries, found {len(commented_all)}"
        )


class TestOpsMainInit:
    """Tests for tileops/ops/__init__.py reduction entries."""

    def test_has_reduction_imports_commented(self):
        content = OPS_INIT.read_text()
        # Should have a commented-out import from .reduction
        found = False
        for line in content.splitlines():
            if ".reduction" in line and line.lstrip().startswith("#"):
                found = True
                break
        assert found, "Commented-out .reduction import not found in ops/__init__.py"

    def test_has_all_op_classes_in_all(self):
        content = OPS_INIT.read_text()
        for cls in OP_CLASSES:
            assert cls in content, f"Op class {cls!r} not found in ops/__init__.py __all__"

    def test_all_op_entries_are_commented(self):
        content = OPS_INIT.read_text()
        for cls in OP_CLASSES:
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Op class {cls!r} should have a commented-out __all__ entry"
            )


class TestRuffLinting:
    """Tests that all files pass ruff check and format."""

    @pytest.mark.parametrize(
        "filepath",
        [KERNEL_INIT, OPS_REDUCTION_INIT, OPS_INIT],
        ids=["kernels/reduction", "ops/reduction", "ops/__init__"],
    )
    def test_ruff_check(self, filepath):
        if not filepath.exists():
            pytest.skip(f"{filepath} does not exist")
        result = subprocess.run(
            ["ruff", "check", str(filepath)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ruff check failed for {filepath}:\n{result.stdout}\n{result.stderr}"
        )

    @pytest.mark.parametrize(
        "filepath",
        [KERNEL_INIT, OPS_REDUCTION_INIT, OPS_INIT],
        ids=["kernels/reduction", "ops/reduction", "ops/__init__"],
    )
    def test_ruff_format(self, filepath):
        if not filepath.exists():
            pytest.skip(f"{filepath} does not exist")
        result = subprocess.run(
            ["ruff", "format", "--check", str(filepath)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"ruff format --check failed for {filepath}:\n{result.stdout}\n{result.stderr}"
        )


class TestImportability:
    """Tests that the init files are importable without errors.

    Uses subprocess to avoid eagerly loading CUDA ops in the test process,
    which can cause side-effect failures in unrelated tests (e.g. fp8).
    """

    def test_kernel_reduction_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tileops.kernels.reduction as m; assert hasattr(m, '__all__')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Failed to import tileops.kernels.reduction:\n{result.stderr}"
        )

    def test_ops_reduction_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tileops.ops.reduction as m; assert hasattr(m, '__all__')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to import tileops.ops.reduction:\n{result.stderr}"

    def test_ops_init_importable(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tileops.ops as m; assert hasattr(m, '__all__')",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed to import tileops.ops:\n{result.stderr}"
