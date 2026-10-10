import math
from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.sequence_modeling.mhc import (
    MHCPostCall,
    MHCPostFwdInterface,
    MHCPostKernel,
    MHCPreCall,
    MHCPreFwdInterface,
    MHCPreKernel,
)
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["MHCPostFwdOp", "MHCPreFwdOp"]


class MHCPreFwdOp(Op):
    """The pre-layer half of Manifold-Constrained Hyper-Connections (mHC).

    Reduces the width-expanded stream to the single tensor a layer consumes, and
    returns the residual that MHCPostFwdOp mixes the layer output back into. The
    expansion width is read from the shape of ``phi`` rather than passed in.

    Layout: BSHD
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"mhc_pre": MHCPreKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"mhc_pre": MHCPreFwdInterface}

    def __init__(
        self,
        alpha_pre: float,
        alpha_post: float,
        alpha_res: float,
        sinkhorn_repeat: int,
        sinkhorn_eps: float = 0.02,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            alpha_pre: Manifest ``params.alpha_pre``, the pre-layer mixing weight.
            alpha_post: Manifest ``params.alpha_post``, the post-layer mixing weight.
            alpha_res: Manifest ``params.alpha_res``, the residual mixing weight.
            sinkhorn_repeat: Manifest ``params.sinkhorn_repeat``, Sinkhorn iterations.
            sinkhorn_eps: Manifest ``params.sinkhorn_eps``, Sinkhorn entropy scale.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.alpha_pre = alpha_pre
        self.alpha_post = alpha_post
        self.alpha_res = alpha_res
        self.sinkhorn_repeat = sinkhorn_repeat
        self.sinkhorn_eps = sinkhorn_eps
        super().__init__(target=target)

    def forward(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            phi: ``[n * c_x, n * n + 2 * n]``, dtype ``float32``.
            x: ``[batch, n * c_x]``, dtype ``bfloat16``.
            b: ``[n * n + 2 * n]``, dtype ``float32``.

        Returns:
            ``x_res``, ``x_layer``, ``h_post``, as the manifest declares.
        """
        n_expand = math.isqrt(phi.shape[1] + 1) - 1
        batch, c_x = x.shape[0], x.shape[1] // n_expand
        phi, x, b = phi.contiguous(), x.contiguous(), b.contiguous()
        call = MHCPreCall(
            batch=batch,
            n_expand=n_expand,
            c_x=c_x,
            dtype=x.dtype,
            alpha_pre=self.alpha_pre,
            alpha_post=self.alpha_post,
            alpha_res=self.alpha_res,
            sinkhorn_repeat=self.sinkhorn_repeat,
            sinkhorn_eps=self.sinkhorn_eps,
            device=x.device,
        )
        kernel = self.kernel_for("mhc_pre", call)
        return kernel(
            phi,
            x,
            b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            self.sinkhorn_repeat,
            self.sinkhorn_eps,
        )

    def roof_key(self) -> str:
        """FLOPs are matmul contractions over the bfloat16 stream; priced on tensor cores."""
        return tensor_core_roof(torch.bfloat16)


class MHCPostFwdOp(Op):
    """The post-layer half of Manifold-Constrained Hyper-Connections (mHC).

    Mixes a layer's output into the residual MHCPreFwdOp set aside, restoring the
    width-expanded stream.

    Layout: BSHD
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"mhc_post": MHCPostKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"mhc_post": MHCPostFwdInterface}

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self, x_layer_out: torch.Tensor, h_post: torch.Tensor, x_res: torch.Tensor
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            x_layer_out: ``[batch, c_x]``, dtype ``bfloat16``.
            h_post: ``[batch, n]``, dtype ``float32``.
            x_res: ``[batch, n * c_x]``, dtype ``bfloat16``.

        Returns:
            ``x_out``, as the manifest declares.
        """
        (batch, c_x), n_expand = x_layer_out.shape, h_post.shape[1]
        call = MHCPostCall(
            batch=batch,
            n_expand=n_expand,
            c_x=c_x,
            dtype=x_layer_out.dtype,
            device=x_layer_out.device,
        )
        inputs = tuple(t.contiguous() for t in (x_layer_out, h_post, x_res))
        kernel = self.kernel_for("mhc_post", call)
        return kernel(*inputs)
