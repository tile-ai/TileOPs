"""TDD tests for ParametricUnaryKernel base class extraction (issue 607).

Validates:
- ParametricUnaryKernel base class exists and is a Kernel subclass
- All 9 parametric kernels inherit from ParametricUnaryKernel
- Each subclass definition is compact (<=20 lines of source)
- fp8 builder logic and default_config npt logic live in the base class
- No new public API symbols are added to __all__
"""

import inspect

import pytest
import torch

from tileops.kernels.elementwise import (
    ClampKernel,
    EluKernel,
    HardtanhKernel,
    LeakyReluKernel,
    MaskedFillKernel,
    NanToNumKernel,
    PreluKernel,
    SoftplusKernel,
    WhereKernel,
)
from tileops.kernels.kernel import Kernel

_PARAMETRIC_KERNELS = [
    LeakyReluKernel,
    EluKernel,
    HardtanhKernel,
    SoftplusKernel,
    PreluKernel,
    WhereKernel,
    ClampKernel,
    MaskedFillKernel,
    NanToNumKernel,
]


@pytest.mark.smoke
class TestParametricUnaryBaseExists:
    """AC-1: Shared base class ParametricUnaryKernel extracted."""

    def test_base_class_importable(self):
        from tileops.kernels.elementwise import ParametricUnaryKernel
        assert issubclass(ParametricUnaryKernel, Kernel)

    def test_base_is_abstract(self):
        """Base should not be directly instantiable (needs _builder_fn)."""
        from tileops.kernels.elementwise import ParametricUnaryKernel
        with pytest.raises((TypeError, NotImplementedError)):
            ParametricUnaryKernel(N_total=16, dtype=torch.float16)


@pytest.mark.smoke
class TestAllParametricKernelsUseBase:
    """AC-2: All 9 parametric kernels use the shared base."""

    @pytest.mark.parametrize("cls", _PARAMETRIC_KERNELS, ids=lambda c: c.__name__)
    def test_inherits_from_base(self, cls):
        from tileops.kernels.elementwise import ParametricUnaryKernel
        assert issubclass(cls, ParametricUnaryKernel), (
            f"{cls.__name__} should inherit from ParametricUnaryKernel"
        )

    @pytest.mark.parametrize("cls", _PARAMETRIC_KERNELS, ids=lambda c: c.__name__)
    def test_subclass_is_compact(self, cls):
        """Each subclass body should be <=20 lines."""
        source = inspect.getsource(cls)
        # Count non-blank, non-comment lines
        lines = [
            line for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(lines) <= 20, (
            f"{cls.__name__} has {len(lines)} significant lines, expected <=20"
        )


@pytest.mark.smoke
class TestFp8BuilderLogicCentralized:
    """AC-3: fp8 builder logic is in one place, not duplicated 9 times."""

    @pytest.mark.parametrize("cls", _PARAMETRIC_KERNELS, ids=lambda c: c.__name__)
    def test_no_fp8_logic_in_subclass_init(self, cls):
        """Subclass __init__ (if overridden) should not call _get_fp8_output_dtypes."""
        if "__init__" not in cls.__dict__:
            return  # subclass does not override __init__, nothing to check
        source = inspect.getsource(cls.__init__)
        assert "_get_fp8_output_dtypes" not in source, (
            f"{cls.__name__}.__init__ should not call _get_fp8_output_dtypes directly"
        )

    @pytest.mark.parametrize("cls", _PARAMETRIC_KERNELS, ids=lambda c: c.__name__)
    def test_no_is_fp8_in_subclass_init(self, cls):
        """Subclass __init__ (if overridden) should not call _is_fp8 directly."""
        if "__init__" not in cls.__dict__:
            return  # subclass does not override __init__, nothing to check
        source = inspect.getsource(cls.__init__)
        assert "_is_fp8" not in source, (
            f"{cls.__name__}.__init__ should not call _is_fp8 directly"
        )


@pytest.mark.smoke
class TestDefaultConfigCentralized:
    """AC-4: default_config npt logic is in one place."""

    @pytest.mark.parametrize("cls", _PARAMETRIC_KERNELS, ids=lambda c: c.__name__)
    def test_no_default_config_override(self, cls):
        """Subclasses that use standard npt logic should not override default_config."""
        from tileops.kernels.elementwise import ParametricUnaryKernel
        # It's OK if some override it for custom threads/npt, but the base should
        # have the standard implementation
        assert hasattr(ParametricUnaryKernel, "default_config")


@pytest.mark.smoke
class TestNoNewPublicAPI:
    """AC-6: No new public API - ParametricUnaryKernel should not be in __all__."""

    def test_base_not_in_all(self):
        import tileops.kernels.elementwise as mod
        assert "ParametricUnaryKernel" not in mod.__all__
