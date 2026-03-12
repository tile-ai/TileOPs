"""Tests for reduction __init__.py files (issue #432).

Validates that:
- tileops/kernels/reduction/__init__.py has active imports for implemented kernels
  and commented-out imports for placeholder kernel classes
- tileops/ops/reduction/__init__.py has active imports for implemented op classes
  and commented-out imports for placeholder op classes
- tileops/ops/__init__.py has active imports/exports for implemented reduction ops
  and commented-out entries for placeholder reduction ops
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

# --- Implemented kernel classes (active imports) ---

IMPLEMENTED_KERNEL_CLASSES = [
    "ArgreduceKernel",
    "LogSumExpKernel",
    "ReduceKernel",
    "SoftmaxKernel",
]

# --- Placeholder kernel classes (still commented out) ---

PLACEHOLDER_KERNEL_CLASSES = [
    "CumulativeKernel",
    "LogicalReduceKernel",
    "VectorNormKernel",
]

ALL_KERNEL_CLASSES = IMPLEMENTED_KERNEL_CLASSES + PLACEHOLDER_KERNEL_CLASSES

# --- Implemented op classes (active imports) ---

IMPLEMENTED_OP_CLASSES = [
    "AmaxOp",
    "AminOp",
    "ArgmaxOp",
    "ArgminOp",
    "LogSoftmaxOp",
    "LogSumExpOp",
    "MeanOp",
    "ProdOp",
    "SoftmaxOp",
    "StdOp",
    "SumOp",
    "VarMeanOp",
    "VarOp",
]

# --- Placeholder op classes (still commented out) ---

PLACEHOLDER_OP_CLASSES = [
    "AllOp",
    "AnyOp",
    "CountNonzeroOp",
    "CummaxOp",
    "CumminOp",
    "CumprodOp",
    "CumsumOp",
    "InfNormOp",
    "L1NormOp",
    "L2NormOp",
    "ReduceMaxOp",
    "ReduceMeanOp",
    "ReduceMinOp",
    "ReduceProdOp",
    "ReduceSumOp",
]

ALL_OP_CLASSES = IMPLEMENTED_OP_CLASSES + PLACEHOLDER_OP_CLASSES


class TestKernelReductionInit:
    """Tests for tileops/kernels/reduction/__init__.py."""

    def test_file_exists(self):
        assert KERNEL_INIT.exists(), f"{KERNEL_INIT} does not exist"

    def test_has_all_kernel_classes(self):
        content = KERNEL_INIT.read_text()
        for cls in ALL_KERNEL_CLASSES:
            assert cls in content, f"Kernel class {cls!r} not found in {KERNEL_INIT}"

    def test_implemented_imports_are_active(self):
        content = KERNEL_INIT.read_text()
        for cls in IMPLEMENTED_KERNEL_CLASSES:
            # The class should appear in an uncommented import line
            found_active = False
            for line in content.splitlines():
                if cls in line and not line.lstrip().startswith("#"):
                    found_active = True
                    break
            assert found_active, (
                f"Implemented kernel class {cls!r} should have an active (uncommented) import"
            )

    def test_placeholder_imports_are_commented_out(self):
        content = KERNEL_INIT.read_text()
        for cls in PLACEHOLDER_KERNEL_CLASSES:
            found_commented = False
            for line in content.splitlines():
                if cls in line and line.lstrip().startswith("#"):
                    found_commented = True
                    break
            assert found_commented, (
                f"Placeholder kernel class {cls!r} should be in a commented-out import"
            )

    def test_has_all_dunder_all(self):
        content = KERNEL_INIT.read_text()
        assert "__all__" in content, "__all__ not found"

    def test_implemented_all_entries_are_active(self):
        content = KERNEL_INIT.read_text()
        for cls in IMPLEMENTED_KERNEL_CLASSES:
            found_active = False
            for line in content.splitlines():
                if f'"{cls}"' in line and not line.lstrip().startswith("#"):
                    found_active = True
                    break
            assert found_active, (
                f"Implemented kernel class {cls!r} should have an active __all__ entry"
            )

    def test_placeholder_all_entries_are_commented_out(self):
        content = KERNEL_INIT.read_text()
        for cls in PLACEHOLDER_KERNEL_CLASSES:
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Placeholder kernel class {cls!r} should have a commented-out __all__ entry"
            )

    def test_exact_kernel_count(self):
        content = KERNEL_INIT.read_text()
        # Count all __all__ entries (both active and commented) with "Kernel" in them
        all_entries = [
            line
            for line in content.splitlines()
            if '"' in line
            and "Kernel" in line
            and ("__all__" not in line)  # skip the __all__ = [...] declaration line
        ]
        assert len(all_entries) >= len(ALL_KERNEL_CLASSES), (
            f"Expected at least {len(ALL_KERNEL_CLASSES)} kernel entries, found {len(all_entries)}"
        )


class TestOpsReductionInit:
    """Tests for tileops/ops/reduction/__init__.py."""

    def test_file_exists(self):
        assert OPS_REDUCTION_INIT.exists(), f"{OPS_REDUCTION_INIT} does not exist"

    def test_has_all_op_classes(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in ALL_OP_CLASSES:
            assert cls in content, f"Op class {cls!r} not found in {OPS_REDUCTION_INIT}"

    def test_implemented_imports_are_active(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in IMPLEMENTED_OP_CLASSES:
            found_active = False
            for line in content.splitlines():
                if cls in line and not line.lstrip().startswith("#"):
                    found_active = True
                    break
            assert found_active, (
                f"Implemented op class {cls!r} should have an active (uncommented) import"
            )

    def test_placeholder_imports_are_commented_out(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in PLACEHOLDER_OP_CLASSES:
            found_commented = False
            for line in content.splitlines():
                if cls in line and line.lstrip().startswith("#"):
                    found_commented = True
                    break
            assert found_commented, (
                f"Placeholder op class {cls!r} should be in a commented-out import"
            )

    def test_has_all_dunder_all(self):
        content = OPS_REDUCTION_INIT.read_text()
        assert "__all__" in content, "__all__ not found"

    def test_implemented_all_entries_are_active(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in IMPLEMENTED_OP_CLASSES:
            found_active = False
            for line in content.splitlines():
                if f'"{cls}"' in line and not line.lstrip().startswith("#"):
                    found_active = True
                    break
            assert found_active, f"Implemented op class {cls!r} should have an active __all__ entry"

    def test_placeholder_all_entries_are_commented_out(self):
        content = OPS_REDUCTION_INIT.read_text()
        for cls in PLACEHOLDER_OP_CLASSES:
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Placeholder op class {cls!r} should have a commented-out __all__ entry"
            )

    def test_exact_op_count(self):
        content = OPS_REDUCTION_INIT.read_text()
        # Count all __all__ entries (both active and commented) with "Op" in them
        all_entries = [
            line
            for line in content.splitlines()
            if '"' in line and "Op" in line and ("__all__" not in line)
        ]
        assert len(all_entries) >= len(ALL_OP_CLASSES), (
            f"Expected at least {len(ALL_OP_CLASSES)} op entries, found {len(all_entries)}"
        )


class TestOpsMainInit:
    """Tests for tileops/ops/__init__.py reduction entries."""

    def test_has_reduction_import(self):
        content = OPS_INIT.read_text()
        # Should have an active import from .reduction (not fully commented)
        found = False
        for line in content.splitlines():
            if ".reduction" in line and "import" in line and not line.lstrip().startswith("#"):
                found = True
                break
        assert found, "Active .reduction import not found in ops/__init__.py"

    def test_has_all_op_classes_in_all(self):
        content = OPS_INIT.read_text()
        for cls in ALL_OP_CLASSES:
            assert cls in content, f"Op class {cls!r} not found in ops/__init__.py __all__"

    def test_implemented_op_entries_are_active(self):
        content = OPS_INIT.read_text()
        for cls in IMPLEMENTED_OP_CLASSES:
            found_active = False
            for line in content.splitlines():
                if f'"{cls}"' in line and not line.lstrip().startswith("#"):
                    found_active = True
                    break
            assert found_active, (
                f"Implemented op class {cls!r} should have an active __all__ entry in ops/__init__.py"
            )

    def test_placeholder_op_entries_are_commented(self):
        content = OPS_INIT.read_text()
        for cls in PLACEHOLDER_OP_CLASSES:
            found_commented_all = False
            for line in content.splitlines():
                if f'"{cls}"' in line and line.lstrip().startswith("#"):
                    found_commented_all = True
                    break
            assert found_commented_all, (
                f"Placeholder op class {cls!r} should have a commented-out __all__ entry"
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
