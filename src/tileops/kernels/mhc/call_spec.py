"""The facts of one Manifold-Constrained Hyper-Connections (mHC) call that its in-tree kernels
select and build on, and the kernel interfaces their implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = ["MHCPostCall", "MHCPostFwdInterface", "MHCPreCall", "MHCPreFwdInterface"]


@dataclasses.dataclass(frozen=True)
class MHCPreCall(CallSpec):
    """One pre-layer mHC step, with the mixing weights the op fixed at construction."""

    batch: int = 0
    n_expand: int = 0
    c_x: int = 0
    dtype: Optional[torch.dtype] = None
    alpha_pre: float = 0.0
    alpha_post: float = 0.0
    alpha_res: float = 0.0
    sinkhorn_repeat: int = 0
    sinkhorn_eps: float = 0.0


@dataclasses.dataclass(frozen=True)
class MHCPostCall(CallSpec):
    """One post-layer mHC step over a stream of ``n_expand`` copies of ``c_x`` channels."""

    batch: int = 0
    n_expand: int = 0
    c_x: int = 0
    dtype: Optional[torch.dtype] = None


class MHCPreFwdInterface(KernelInterface):
    """The mHC pre-layer half: reduce the width-expanded stream to what one layer consumes."""

    request = MHCPreCall

    @abstractmethod
    def forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        b: torch.Tensor,
        alpha_pre: float,
        alpha_post: float,
        alpha_res: float,
        sinkhorn_repeat: int,
        sinkhorn_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix the expanded stream down to one layer input; nothing is written in place.

        Every tensor is contiguous on ``call.device``. The five weights repeat
        ``call.alpha_pre`` through ``call.sinkhorn_eps`` at the launch, because the program
        takes them as scalar arguments and none of them changes what is built.

        Args:
            phi: ``float32`` ``(n_expand * c_x, n_expand * n_expand + 2 * n_expand)``.
            x: ``(batch, n_expand * c_x)`` in ``call.dtype``.
            b: ``float32`` ``(n_expand * n_expand + 2 * n_expand,)``.
            alpha_pre: The pre-layer mixing weight.
            alpha_post: The post-layer mixing weight.
            alpha_res: The residual mixing weight.
            sinkhorn_repeat: Sinkhorn iterations.
            sinkhorn_eps: Sinkhorn entropy scale.

        Returns:
            New ``(x_res, x_layer, h_post)``: ``x_res`` shaped like *x* in ``call.dtype``,
            ``x_layer`` ``(batch, c_x)`` in ``call.dtype``, and ``float32``
            ``(batch, n_expand)`` weights for the post-layer half.
        """


class MHCPostFwdInterface(KernelInterface):
    """The mHC post-layer half: mix a layer's output back into the width-expanded stream."""

    request = MHCPostCall

    @abstractmethod
    def forward(
        self, x_layer_out: torch.Tensor, h_post: torch.Tensor, x_res: torch.Tensor
    ) -> torch.Tensor:
        """Add the layer output back into the residual; nothing is written in place.

        Every tensor is contiguous on ``call.device``, and all three are what the pre-layer
        half set aside and the layer produced.

        Args:
            x_layer_out: ``(batch, c_x)`` layer output in ``call.dtype``.
            h_post: ``float32`` ``(batch, n_expand)`` post-layer weights.
            x_res: ``(batch, n_expand * c_x)`` residual in ``call.dtype``.

        Returns:
            A new ``(batch, n_expand * c_x)`` stream in ``call.dtype``.
        """
