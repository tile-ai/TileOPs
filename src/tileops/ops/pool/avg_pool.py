from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.pool import (
    AvgPool1dFwdInterface,
    AvgPool1dKernel,
    AvgPool1dRegisterKernel,
    AvgPool2dFwdInterface,
    AvgPool2dKernel,
    AvgPool2dRegisterKernel,
    AvgPool3dFwdInterface,
    AvgPool3dKernel,
    AvgPoolCall,
)
from tileops.ops.op_base import Op
from tileops.ops.pool.parameters import _per_axis

__all__ = ["AvgPool1dFwdOp", "AvgPool2dFwdOp", "AvgPool3dFwdOp"]


class _AvgPoolFwdOpBase(Op):
    """Generic average-pooling forward, parametrized by class-attribute ``ndim``.

    Concrete subclasses set ``ndim``, ``kernel_types`` and ``interfaces`` and state the
    manifest's ``__init__``; the signature's checks, shape inference and roofline are generated per
    concrete class.
    """

    ndim: ClassVar[int]

    def _setup(self, kernel_map: Optional[Dict[str, Kernel]]) -> None:
        """Resolve the kernel map, then the per-axis parameters the kernels take."""
        self.dispatch_kernel(kernel_map)
        nd = self.ndim
        self._kernel_size = _per_axis(self.kernel_size, nd)
        self._stride = self._kernel_size if self.stride is None else _per_axis(self.stride, nd)
        self._padding = _per_axis(self.padding, nd)
        self._divisor_override = getattr(self, "divisor_override", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        input = input.contiguous()
        n, c_in, *in_dims = input.shape
        call = AvgPoolCall(
            n=n,
            c_in=c_in,
            size=tuple(in_dims),
            window=self._kernel_size,
            stride=self._stride,
            pad=self._padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self._divisor_override,
            dtype=input.dtype,
            device=input.device,
        )
        kernel = self.kernel_for("avg_pool", call)
        return kernel(input)


class AvgPool1dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCL inputs."""

    ndim = 1
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "avg_pool1d_kernel": AvgPool1dKernel,
        "avg_pool1d_register": AvgPool1dRegisterKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"avg_pool": AvgPool1dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        # No divisor_override: torch.nn.functional.avg_pool1d does not take one.
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.target = target
        self.tune = tune
        self._setup(kernel_map)


class AvgPool2dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCHW inputs."""

    ndim = 2
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "avg_pool2d_kernel": AvgPool2dKernel,
        "avg_pool2d_register": AvgPool2dRegisterKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"avg_pool": AvgPool2dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            divisor_override: Manifest ``params.divisor_override``, ``int | None``, default ``None``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.target = target
        self.tune = tune
        self._setup(kernel_map)


class AvgPool3dFwdOp(_AvgPoolFwdOpBase):
    """Average pooling over PyTorch-compatible NCDHW inputs."""

    ndim = 3
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"avg_pool3d_kernel": AvgPool3dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"avg_pool": AvgPool3dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int, int, int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int, int, int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int, int, int]``, default ``0``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            count_include_pad: Manifest ``params.count_include_pad``, ``bool``, default ``True``.
            divisor_override: Manifest ``params.divisor_override``, ``int | None``, default ``None``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.target = target
        self.tune = tune
        self._setup(kernel_map)
