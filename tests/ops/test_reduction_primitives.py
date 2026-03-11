"""Tests for tileops/kernels/reduction/_primitives.py.

Validates the shared reduction primitives: utility functions, constants,
and T.macro factory functions.
"""

import pytest


class TestAlignUp:
    """Tests for _align_up utility function."""

    def test_already_aligned(self):
        from tileops.kernels.reduction._primitives import _align_up

        assert _align_up(256, 256) == 256

    def test_needs_padding(self):
        from tileops.kernels.reduction._primitives import _align_up

        assert _align_up(100, 256) == 256

    def test_one_over(self):
        from tileops.kernels.reduction._primitives import _align_up

        assert _align_up(257, 256) == 512

    def test_zero(self):
        from tileops.kernels.reduction._primitives import _align_up

        assert _align_up(0, 256) == 0

    def test_custom_alignment(self):
        from tileops.kernels.reduction._primitives import _align_up

        assert _align_up(10, 8) == 16
        assert _align_up(8, 8) == 8
        assert _align_up(9, 8) == 16


class TestDefaultAlignment:
    """Tests for DEFAULT_ALIGNMENT constant."""

    def test_value(self):
        from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT

        assert DEFAULT_ALIGNMENT == 256

    def test_is_int(self):
        from tileops.kernels.reduction._primitives import DEFAULT_ALIGNMENT

        assert isinstance(DEFAULT_ALIGNMENT, int)


class TestMakeReduceEpilogue:
    """Tests for make_reduce_epilogue factory."""

    def test_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        macro = make_reduce_epilogue("sum")
        assert callable(macro)

    def test_sum_kind(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        macro = make_reduce_epilogue("sum")
        assert macro is not None

    def test_max_kind(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        macro = make_reduce_epilogue("max")
        assert macro is not None

    def test_min_kind(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        macro = make_reduce_epilogue("min")
        assert macro is not None

    def test_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_reduce_epilogue

        with pytest.raises(ValueError, match="Unsupported"):
            make_reduce_epilogue("invalid_op")


class TestMakeWelfordUpdate:
    """Tests for make_welford_update factory."""

    def test_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_welford_update

        macro = make_welford_update(block_m=4, N_padded=256)
        assert callable(macro)

    def test_different_shapes(self):
        from tileops.kernels.reduction._primitives import make_welford_update

        macro1 = make_welford_update(block_m=4, N_padded=256)
        macro2 = make_welford_update(block_m=8, N_padded=512)
        assert macro1 is not None
        assert macro2 is not None


class TestMakeSoftmaxEpilogue:
    """Tests for make_softmax_epilogue factory."""

    def test_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        macro = make_softmax_epilogue("softmax")
        assert callable(macro)

    def test_softmax_kind(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        macro = make_softmax_epilogue("softmax")
        assert macro is not None

    def test_log_softmax_kind(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        macro = make_softmax_epilogue("log_softmax")
        assert macro is not None

    def test_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_softmax_epilogue

        with pytest.raises(ValueError, match="Unsupported"):
            make_softmax_epilogue("invalid_op")


class TestMakeCumulativeScan:
    """Tests for make_cumulative_scan factory."""

    def test_returns_callable(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        macro = make_cumulative_scan("sum")
        assert callable(macro)

    def test_sum_kind(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        macro = make_cumulative_scan("sum")
        assert macro is not None

    def test_prod_kind(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        macro = make_cumulative_scan("prod")
        assert macro is not None

    def test_invalid_kind_raises(self):
        from tileops.kernels.reduction._primitives import make_cumulative_scan

        with pytest.raises(ValueError, match="Unsupported"):
            make_cumulative_scan("invalid_op")


class TestInitReExports:
    """Tests for __init__.py re-exports (AC-2)."""

    def test_kernel_init_all(self):
        import tileops.kernels.reduction as reduction

        assert hasattr(reduction, "__all__")
        expected = [
            "_align_up",
            "DEFAULT_ALIGNMENT",
            "make_reduce_epilogue",
            "make_welford_update",
            "make_softmax_epilogue",
            "make_cumulative_scan",
        ]
        for name in expected:
            assert name in reduction.__all__, f"{name} missing from __all__"

    def test_kernel_init_imports(self):
        from tileops.kernels.reduction import (
            DEFAULT_ALIGNMENT,
            _align_up,
            make_cumulative_scan,
            make_reduce_epilogue,
            make_softmax_epilogue,
            make_welford_update,
        )

        assert _align_up is not None
        assert DEFAULT_ALIGNMENT == 256
        assert callable(make_reduce_epilogue)
        assert callable(make_welford_update)
        assert callable(make_softmax_epilogue)
        assert callable(make_cumulative_scan)

    def test_ops_init_all(self):
        import tileops.ops.reduction as reduction

        assert hasattr(reduction, "__all__")
        assert isinstance(reduction.__all__, list)
