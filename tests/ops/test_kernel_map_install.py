"""Tests for the shared ``_install_kernel_map`` validation path.

A user-supplied ``kernel_map`` and the auto-discovered ``default_kernel_map``
must traverse the same validate-and-install path in the Op base, so that
architecture-compatibility checks fire identically regardless of provenance.
"""


import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version


def _make_incompatible_arch_list() -> list[int]:
    """Return a ``supported_archs`` list that excludes the current device."""
    current = get_sm_version()
    candidates = [70, 75, 80, 86, 89, 90, 100]
    incompatible = [a for a in candidates if a != current]
    assert incompatible, "no incompatible arch candidate available"
    return incompatible


@pytest.mark.smoke
def test_install_kernel_map_user_supplied_incompatible_raises_valueerror() -> None:
    """User-supplied incompatible kernel raises ``ValueError`` (auto-discovery class)."""
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    inst = cls(N_total=8, dtype=torch.float16)
    (key, default_kernel_cls), = inst.default_kernel_map.items()
    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = incompatible_archs

    with pytest.raises(ValueError, match="not supported on architecture"):
        cls(N_total=8, dtype=torch.float16, kernel_map={key: IncompatibleKernel})


@pytest.mark.smoke
def test_install_kernel_map_auto_discovery_incompatible_raises_same_class() -> None:
    """Auto-discovery path raises the same ``ValueError`` on identical input.

    Build an Op subclass whose ``default_kernel_map`` already points at an
    arch-incompatible kernel; constructing it must raise the same class
    that the user-supplied path produces.
    """
    import tileops.ops.elementwise as mod

    base_cls = mod.ReluFwdOp
    base_inst = base_cls(N_total=8, dtype=torch.float16)
    (key, default_kernel_cls), = base_inst.default_kernel_map.items()
    incompatible_archs = _make_incompatible_arch_list()

    class IncompatibleKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = incompatible_archs

    class AutoDiscoveredIncompatibleOp(base_cls):  # type: ignore[misc, valid-type]
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {key: IncompatibleKernel}

    with pytest.raises(ValueError, match="not supported on architecture"):
        AutoDiscoveredIncompatibleOp(N_total=8, dtype=torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.smoke
def test_install_kernel_map_compatible_override_forward_bit_identical() -> None:
    """A compatible user-supplied override yields bit-identical forward output.

    Build an op with the default kernel and the same op with a marker
    subclass override (same kernel logic, distinct identity). Both must
    produce the exact same forward output on identical input.
    """
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    n_total = 128
    dtype = torch.float16

    baseline = cls(N_total=n_total, dtype=dtype)
    (key, default_kernel_cls), = baseline.default_kernel_map.items()

    class MarkerKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Subclass marker; identical behavior, distinct identity."""

    overridden = cls(
        N_total=n_total, dtype=dtype, kernel_map={key: MarkerKernel},
    )
    assert isinstance(overridden.kernel, MarkerKernel)

    torch.manual_seed(0)
    x = torch.randn(n_total, dtype=dtype, device="cuda")
    y_baseline = baseline(x.clone())
    y_overridden = overridden(x.clone())
    assert torch.equal(y_baseline, y_overridden), (
        "compatible kernel_map override must yield bit-identical forward output"
    )


@pytest.mark.smoke
def test_install_kernel_map_supported_archs_none() -> None:
    """A kernel with ``supported_archs=None`` installs without raising.

    The base ``Kernel`` class declares ``supported_archs: Optional[list[int]]``
    defaulting to ``None``. The validate-and-install path must treat ``None``
    as "no arch restriction" rather than attempting ``in None``, which would
    raise ``TypeError``.
    """
    import tileops.ops.elementwise as mod

    cls = mod.ReluFwdOp
    inst = cls(N_total=8, dtype=torch.float16)
    (key, default_kernel_cls), = inst.default_kernel_map.items()

    class UnrestrictedKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        supported_archs = None

    overridden = cls(N_total=8, dtype=torch.float16, kernel_map={key: UnrestrictedKernel})
    assert overridden.kernel_map[key] is UnrestrictedKernel


@pytest.mark.smoke
def test_install_kernel_map_is_private_helper_only() -> None:
    """Refactor exposes ``_install_kernel_map`` only — no new public API.

    Guards AC: the shared path is private (leading underscore); the public
    surface is unchanged (``dispatch_kernel``, ``kernel_map``, ``tune``).
    """
    from tileops.ops.op_base import Op

    assert hasattr(Op, "_install_kernel_map")
    assert hasattr(Op, "dispatch_kernel")
    public_names = [n for n in vars(Op) if not n.startswith("_")]
    assert "install_kernel_map" not in public_names
