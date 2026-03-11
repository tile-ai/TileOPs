"""Tests for tileops/kernels/reduction/_primitives.py.

Validates the shared reduction primitives: utility functions, constants,
and T.macro factory functions.  Tests verify both API contracts (callability,
error handling) and correctness of the generated macro logic.
"""

import pytest


# -----------------------------------------------------------------------
# align_up
# -----------------------------------------------------------------------
class TestAlignUp:
    """Tests for align_up utility function."""

    @pytest.mark.smoke
    def test_align_up_already_aligned(self):
        from tileops.kernels.reduction._primitives import align_up

        assert align_up(256, 256) == 256

    @pytest.mark.smoke
    def test_align_up_needs_padding(self):
        from tileops.kernels.reduction._primitives import align_up

        assert align_up(100, 256) == 256

    @pytest.mark.smoke
    def test_align_up_one_over(self):
        from tileops.kernels.reduction._primitives import align_up

        assert align_up(257, 256) == 512

    @pytest.mark.smoke
    def test_align_up_zero(self):
        from tileops.kernels.reduction._primitives import align_up

        assert align_up(0, 256) == 0

    @pytest.mark.smoke
    def test_align_up_custom_alignment(self):
        from tileops.kernels.reduction._primitives import align_up

        assert align_up(10, 8) == 16
        assert align_up(8, 8) == 8
        assert align_up(9, 8) == 16

    @pytest.mark.smoke
    def test_align_up_non_positive_raises(self):
        from tileops.kernels.reduction._primitives import align_up

        with pytest.raises(ValueError, match="positive"):
            align_up(10, 0)
        with pytest.raises(ValueError, match="positive"):
            align_up(10, -1)

    @pytest.mark.smoke
    def test_align_up_powers_of_two(self):
        """Verify correctness across a range of power-of-two alignments."""
        from tileops.kernels.reduction._primitives import align_up

        for p in range(1, 12):
            alignment = 2**p
            assert align_up(alignment - 1, alignment) == alignment
            assert align_up(alignment, alignment) == alignment
            assert align_up(alignment + 1, alignment) == 2 * alignment


# -----------------------------------------------------------------------
# DEFAULT_ALIGNMENT
# -----------------------------------------------------------------------
class TestDefaultAlignment:
    """Tests for DEFAULT_ALIGNMENT constant."""

    @pytest.mark.smoke
    def test_default_alignment_value(self):
        from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT

        assert DEFAULT_ALIGNMENT == 256

    @pytest.mark.smoke
    def test_default_alignment_is_int(self):
        from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT

        assert isinstance(DEFAULT_ALIGNMENT, int)


# -----------------------------------------------------------------------
# make_reduce_epilogue
# -----------------------------------------------------------------------
class TestMakeReduceEpilogue:
    """Tests for make_reduce_epilogue factory."""

    @pytest.mark.smoke
    def test_reduce_epilogue_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        macro = make_reduce_epilogue("sum")
        assert callable(macro)

    @pytest.mark.smoke
    def test_reduce_epilogue_all_valid_kinds(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        for kind in ("sum", "max", "min"):
            macro = make_reduce_epilogue(kind)
            assert callable(macro)

    @pytest.mark.smoke
    def test_reduce_epilogue_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        with pytest.raises(ValueError, match="Unsupported"):
            make_reduce_epilogue("invalid_op")


# -----------------------------------------------------------------------
# make_welford_update
# -----------------------------------------------------------------------
class TestMakeWelfordUpdate:
    """Tests for make_welford_update factory."""

    @pytest.mark.smoke
    def test_welford_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_welford_update

        macro = make_welford_update(block_m=4, N_padded=256)
        assert callable(macro)

    @pytest.mark.smoke
    def test_welford_different_shapes(self):
        from tileops.kernels.reduction._primitives import make_welford_update

        m1 = make_welford_update(block_m=4, N_padded=256)
        m2 = make_welford_update(block_m=8, N_padded=512)
        assert m1 is not None
        assert m2 is not None

    @pytest.mark.smoke
    def test_welford_uses_reduce_sum(self):
        """Verify the macro uses T.reduce_sum for parallel-safe reduction."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_welford_update)
        assert "T.reduce_sum" in src, "Welford must use T.reduce_sum for parallel-safe reduction"

    @pytest.mark.smoke
    def test_welford_no_race_condition(self):
        """The source must NOT update mean[i] inside a T.Parallel(block_m, N_padded) loop."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_welford_update)
        assert "mean[i] = mean[i] + delta" not in src, "Found racy mean update inside parallel loop"

    @pytest.mark.smoke
    def test_welford_sq_diff_uses_batch_mean(self):
        """sq_diff must use batch_mean (batch's own mean), not new_mean (combined mean).

        The parallel Welford merge formula requires M2_b = sum((x[j] - mean_b)^2)
        where mean_b is the batch's own mean, not the combined mean of old + batch.
        """
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_welford_update)
        assert "batch_mean" in src, "Welford must compute batch_mean for M2_b calculation"
        assert "x[i, j] - batch_mean[i]" in src, "sq_diff must use batch_mean[i], not new_mean[i]"
        # Ensure sq_diff does NOT use new_mean
        assert "x[i, j] - new_mean[i]" not in src, (
            "sq_diff must NOT use new_mean (combined mean) -- causes incorrect variance"
        )


# -----------------------------------------------------------------------
# make_softmax_epilogue
# -----------------------------------------------------------------------
class TestMakeSoftmaxEpilogue:
    """Tests for make_softmax_epilogue factory."""

    @pytest.mark.smoke
    def test_softmax_epilogue_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        macro = make_softmax_epilogue("softmax")
        assert callable(macro)

    @pytest.mark.smoke
    def test_log_softmax_epilogue_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        macro = make_softmax_epilogue("log_softmax")
        assert callable(macro)

    @pytest.mark.smoke
    def test_softmax_epilogue_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        with pytest.raises(ValueError, match="Unsupported"):
            make_softmax_epilogue("invalid_op")

    @pytest.mark.smoke
    def test_softmax_epilogue_has_division(self):
        """Softmax epilogue must normalize by row_sum (division)."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_softmax_epilogue)
        assert "/ row_sum[i]" in src, "Softmax epilogue must divide by row_sum"

    @pytest.mark.smoke
    def test_log_softmax_epilogue_has_log(self):
        """Log-softmax epilogue must apply T.log."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_softmax_epilogue)
        assert "T.log" in src, "Log-softmax epilogue must use T.log"

    @pytest.mark.smoke
    def test_softmax_vs_log_softmax_differ(self):
        """The two softmax variants must produce different macros."""
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        m_soft = make_softmax_epilogue("softmax")
        m_log = make_softmax_epilogue("log_softmax")
        assert m_soft is not m_log


# -----------------------------------------------------------------------
# make_cumulative_scan
# -----------------------------------------------------------------------
class TestMakeCumulativeScan:
    """Tests for make_cumulative_scan factory."""

    @pytest.mark.smoke
    def test_cumsum_scan_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        macro = make_cumulative_scan("sum")
        assert callable(macro)

    @pytest.mark.smoke
    def test_cumprod_scan_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        macro = make_cumulative_scan("prod")
        assert callable(macro)

    @pytest.mark.smoke
    def test_cumulative_scan_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        with pytest.raises(ValueError, match="Unsupported"):
            make_cumulative_scan("invalid_op")

    @pytest.mark.smoke
    def test_cumulative_scan_uses_serial_loop(self):
        """Cumulative scan must use T.Serial for sequential dependency."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_cumulative_scan)
        assert "T.Serial" in src, "Cumulative scan must use T.Serial for sequential scan"

    @pytest.mark.smoke
    def test_cumsum_scan_has_addition(self):
        """Sum scan must accumulate via addition."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_cumulative_scan)
        assert "output_buf[i, j - 1] + input_buf[i, j]" in src, (
            "Sum scan must add previous output to current input"
        )

    @pytest.mark.smoke
    def test_cumprod_scan_has_multiplication(self):
        """Prod scan must accumulate via multiplication."""
        import inspect

        import tileops.kernels.reduction._primitives as mod

        src = inspect.getsource(mod.make_cumulative_scan)
        assert "output_buf[i, j - 1] * input_buf[i, j]" in src, (
            "Prod scan must multiply previous output by current input"
        )

    @pytest.mark.smoke
    def test_cumsum_vs_cumprod_differ(self):
        """Sum and prod scans must be different macros."""
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        m_sum = make_cumulative_scan("sum")
        m_prod = make_cumulative_scan("prod")
        assert m_sum is not m_prod


# -----------------------------------------------------------------------
# __init__.py re-exports (AC-2)
# -----------------------------------------------------------------------
class TestInitReExports:
    """Tests for __init__.py re-exports (AC-2)."""

    @pytest.mark.smoke
    def test_kernel_init_has_all(self):
        import tileops.kernels.reduction as reduction

        assert hasattr(reduction, "__all__")
        expected = [
            "align_up",
            "DEFAULT_ALIGNMENT",
            "make_reduce_epilogue",
            "make_welford_update",
            "make_softmax_epilogue",
            "make_cumulative_scan",
        ]
        for name in expected:
            assert name in reduction.__all__, f"{name} missing from __all__"

    @pytest.mark.smoke
    def test_kernel_init_imports_work(self):
        from tileops.kernels.reduction import (
            DEFAULT_ALIGNMENT,
            align_up,
            make_cumulative_scan,
            make_reduce_epilogue,
            make_softmax_epilogue,
            make_welford_update,
        )

        assert align_up is not None
        assert DEFAULT_ALIGNMENT == 256
        assert callable(make_reduce_epilogue)
        assert callable(make_welford_update)
        assert callable(make_softmax_epilogue)
        assert callable(make_cumulative_scan)

    @pytest.mark.smoke
    def test_kernel_init_no_underscore_in_all(self):
        """Public __all__ should not export underscore-prefixed names."""
        import tileops.kernels.reduction as reduction

        for name in reduction.__all__:
            assert not name.startswith("_"), f"'{name}' has underscore prefix but is in __all__"

    @pytest.mark.smoke
    def test_ops_init_has_all(self):
        import tileops.ops.reduction as reduction

        assert hasattr(reduction, "__all__")
        assert isinstance(reduction.__all__, list)
