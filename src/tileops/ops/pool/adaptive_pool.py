from math import prod
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.pool import (
    AdaptiveAvgPool2dFwdInterface,
    AdaptiveAvgPool2dKernel,
    AdaptiveMaxPool2dFwdInterface,
    AdaptiveMaxPool2dIndicesFwdInterface,
    AdaptiveMaxPool2dKernel,
    AdaptiveMaxPool2dWithIndicesKernel,
    AdaptivePool2dCall,
)
from tileops.ops.op_base import Op

__all__ = ["AdaptiveAvgPool2dFwdOp", "AdaptiveMaxPool2dFwdOp", "AdaptiveMaxPool2dIndicesFwdOp"]


class _AdaptivePool2dFwdOpBase(Op):
    """Generic adaptive 2D pooling forward over CHW/NCHW inputs.

    Concrete subclasses set ``kernel_types`` and ``interfaces`` and state the manifest's
    ``__init__``. A CHW input is handed to the kernel as it is; the kernel adds and drops
    the batch axis.
    """

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``;
                a ``None`` extent keeps the input's.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.output_size = output_size
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._output_size = (
            (output_size, output_size)
            if output_size is None or isinstance(output_size, int)
            else tuple(output_size)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on ``input``."""
        return self._call_boundary(input)

    def _eager_forward(self, input: torch.Tensor):
        input = input.contiguous()
        c_in, h_in, w_in = input.shape[-3:]
        call = AdaptivePool2dCall(
            n=prod(input.shape[:-3]),
            c_in=c_in,
            h_in=h_in,
            w_in=w_in,
            out_h=h_in if self._output_size[0] is None else self._output_size[0],
            out_w=w_in if self._output_size[1] is None else self._output_size[1],
            dtype=input.dtype,
            device=input.device,
        )
        self.kernel = self.kernel_for("adaptive_pool", call)
        return self.kernel(input)


class AdaptiveAvgPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive average pooling over PyTorch-compatible CHW/NCHW inputs."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_avg_pool2d_kernel": AdaptiveAvgPool2dKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "adaptive_pool": AdaptiveAvgPool2dFwdInterface
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)


class AdaptiveMaxPool2dFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=False)."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_max_pool2d_kernel": AdaptiveMaxPool2dKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "adaptive_pool": AdaptiveMaxPool2dFwdInterface
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)


class AdaptiveMaxPool2dIndicesFwdOp(_AdaptivePool2dFwdOpBase):
    """Adaptive max pooling over CHW/NCHW inputs (return_indices=True)."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "adaptive_max_pool2d_with_indices_kernel": AdaptiveMaxPool2dWithIndicesKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "adaptive_pool": AdaptiveMaxPool2dIndicesFwdInterface
    }

    def __init__(
        self,
        output_size: int | None | Tuple[Optional[int], Optional[int]] | list[Optional[int]],
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            output_size: Manifest ``params.output_size``, ``int | None | tuple[int | None, int | None] | list[int | None]``.
            target: Backend target to serve this op, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        super().__init__(output_size=output_size, target=target, kernel_map=kernel_map, tune=tune)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16``.

        Returns:
            ``output``, ``indices``, as the manifest declares.
        """
        return self._call_boundary(input)
