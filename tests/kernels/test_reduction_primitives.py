"""Tests for src/tileops/kernels/reduction/_primitives.py."""

import pytest

# align_up


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


# DEFAULT_ALIGNMENT


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


class TestInitReExports:
    """Tests for __init__.py re-exports."""

    @pytest.mark.smoke
    def test_kernel_init_has_all(self):
        import tileops.kernels.reduction as reduction

        assert hasattr(reduction, "__all__")
        for name in ("align_up", "DEFAULT_ALIGNMENT"):
            assert name in reduction.__all__, f"{name} missing from __all__"

    @pytest.mark.smoke
    def test_kernel_init_imports_work(self):
        from tileops.kernels.reduction import DEFAULT_ALIGNMENT, align_up

        assert callable(align_up)
        assert DEFAULT_ALIGNMENT == 256

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
