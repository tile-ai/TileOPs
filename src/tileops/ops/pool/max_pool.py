from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.pool import (
    MaxPool1dFwdInterface,
    MaxPool1dIndicesFwdInterface,
    MaxPool1dKernel,
    MaxPool1dWithIndicesKernel,
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPool2dKernel,
    MaxPool2dWithIndicesKernel,
    MaxPool3dFwdInterface,
    MaxPool3dIndicesFwdInterface,
    MaxPool3dKernel,
    MaxPool3dWithIndicesKernel,
    MaxPoolCall,
)
from tileops.ops.op_base import Op
from tileops.ops.pool.parameters import _per_axis

__all__ = [
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
    "MaxPool3dFwdOp",
    "MaxPool3dIndicesFwdOp",
]


class _MaxPoolFwdOpBase(Op):
    """Generic max-pooling forward, parametrized by class attributes.

    Concrete subclasses set ``ndim``, ``kernel_types`` and ``interfaces`` and state the
    manifest's ``__init__``.
    """

    ndim: ClassVar[int]
    compile_boundary = True

    def __init__(
        self,
        kernel_size: "int | Tuple[int, ...]",
        stride: "Optional[int | Tuple[int, ...]]" = None,
        padding: "int | Tuple[int, ...]" = 0,
        dilation: "int | Tuple[int, ...]" = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, an int or one per spatial axis.
            stride: Manifest ``params.stride``, an int, one per axis, or ``None``.
            padding: Manifest ``params.padding``, an int or one per spatial axis.
            dilation: Manifest ``params.dilation``, an int or one per spatial axis.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        nd = self.ndim
        self._kernel_size = _per_axis(kernel_size, nd)
        self._stride = self._kernel_size if stride is None else _per_axis(stride, nd)
        self._padding = _per_axis(padding, nd)
        self._dilation = _per_axis(dilation, nd)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)

    def _eager_forward(self, input: torch.Tensor):
        input = input.contiguous()
        n, c_in, *in_dims = input.shape
        call = MaxPoolCall(
            n=n,
            c_in=c_in,
            size=tuple(in_dims),
            window=self._kernel_size,
            stride=self._stride,
            pad=self._padding,
            dilation=self._dilation,
            ceil_mode=self.ceil_mode,
            dtype=input.dtype,
            device=input.device,
        )
        self.kernel = self.kernel_for("max_pool", call)
        return self.kernel(input)


class MaxPool1dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=False)."""

    ndim = 1
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool1d_kernel": MaxPool1dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"max_pool": MaxPool1dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


class MaxPool1dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCL inputs (return_indices=True)."""

    ndim = 1
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool1d_with_indices_kernel": MaxPool1dWithIndicesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "max_pool": MaxPool1dIndicesFwdInterface
    }

    def __init__(
        self,
        kernel_size: int | Tuple[int],
        stride: Optional[int | Tuple[int]] = None,
        padding: int | Tuple[int] = 0,
        dilation: int | Tuple[int] = 1,
        ceil_mode: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_size: Manifest ``params.kernel_size``, ``int | tuple[int]``.
            stride: Manifest ``params.stride``, ``int | tuple[int] | None``, default ``None``.
            padding: Manifest ``params.padding``, ``int | tuple[int]``, default ``0``.
            dilation: Manifest ``params.dilation``, ``int | tuple[int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)


class MaxPool2dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=False)."""

    ndim = 2
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool2d_kernel": MaxPool2dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"max_pool": MaxPool2dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
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
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


class MaxPool2dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCHW inputs (return_indices=True)."""

    ndim = 2
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool2d_with_indices_kernel": MaxPool2dWithIndicesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "max_pool": MaxPool2dIndicesFwdInterface
    }

    def __init__(
        self,
        kernel_size: int | Tuple[int, int],
        stride: Optional[int | Tuple[int, int]] = None,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        ceil_mode: bool = False,
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
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)


class MaxPool3dFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=False)."""

    ndim = 3
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"max_pool3d_kernel": MaxPool3dKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"max_pool": MaxPool3dFwdInterface}

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        ceil_mode: bool = False,
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
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )


class MaxPool3dIndicesFwdOp(_MaxPoolFwdOpBase):
    """Max pooling over PyTorch-compatible NCDHW inputs (return_indices=True)."""

    ndim = 3
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "max_pool3d_with_indices_kernel": MaxPool3dWithIndicesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "max_pool": MaxPool3dIndicesFwdInterface
    }

    def __init__(
        self,
        kernel_size: int | Tuple[int, int, int],
        stride: Optional[int | Tuple[int, int, int]] = None,
        padding: int | Tuple[int, int, int] = 0,
        dilation: int | Tuple[int, int, int] = 1,
        ceil_mode: bool = False,
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
            dilation: Manifest ``params.dilation``, ``int | tuple[int, int, int]``, default ``1``.
            ceil_mode: Manifest ``params.ceil_mode``, ``bool``, default ``False``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            target=target,
            kernel_map=kernel_map,
            tune=tune,
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)
