import math
from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mhc import MHCPostKernel, MHCPreKernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["MHCPostFwdOp", "MHCPreFwdOp"]


class MHCPreFwdOp(Op):
    """The pre-layer half of Manifold-Constrained Hyper-Connections (mHC).

    Reduces the width-expanded stream to the single tensor a layer consumes, and
    returns the residual that MHCPostFwdOp mixes the layer output back into. The
    expansion width is read from the shape of ``phi`` rather than passed in.

    Layout: BSHD
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"mhc_pre_kernel": MHCPreKernel}

    def __init__(
        self,
        alpha_pre: float,
        alpha_post: float,
        alpha_res: float,
        sinkhorn_repeat: int,
        sinkhorn_eps: float = 0.02,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.alpha_pre = alpha_pre
        self.alpha_post = alpha_post
        self.alpha_res = alpha_res
        self.sinkhorn_repeat = sinkhorn_repeat
        self.sinkhorn_eps = sinkhorn_eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        batch, n_expand, c_x, dtype, _device = call
        return call, lambda: self.kernel_map["mhc_pre_kernel"](
            batch, n_expand, c_x, dtype, tune=self.tune
        )

    def forward(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            phi: ``[n * c_x, n * n + 2 * n]``, dtype ``float32``.
            x: ``[batch, n * c_x]``, dtype ``bfloat16``.
            b: ``[n * n + 2 * n]``, dtype ``float32``.

        Returns:
            ``x_res``, ``x_layer``, ``h_post``, as the manifest declares.
        """
        return self._call_boundary(phi, x, b)

    def _eager_forward(
        self,
        phi: torch.Tensor,
        x: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        # The call check has solved n * n + 2 * n == phi.shape[1] and n | x.shape[1].
        n_expand = math.isqrt(phi.shape[1] + 1) - 1
        batch, c_x = x.shape[0], x.shape[1] // n_expand
        phi, x, b = phi.contiguous(), x.contiguous(), b.contiguous()
        key = (batch, n_expand, c_x, x.dtype, x.device.index)
        self.kernel = self.kernel_for("mhc_pre_kernel", (phi, x, b), key)
        return self.kernel(
            phi,
            x,
            b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            self.sinkhorn_repeat,
            self.sinkhorn_eps,
        )

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions over the bfloat16 stream; priced on tensor cores."""
        return tensor_core_roof(torch.bfloat16)


class MHCPostFwdOp(Op):
    """The post-layer half of Manifold-Constrained Hyper-Connections (mHC).

    Mixes a layer's output into the residual MHCPreFwdOp set aside, restoring the
    width-expanded stream.

    Layout: BSHD
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"mhc_post_kernel": MHCPostKernel}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        batch, n_expand, c_x, dtype, _device = call
        return call, lambda: self.kernel_map["mhc_post_kernel"](
            batch, n_expand, c_x, dtype, tune=self.tune
        )

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
        return self._call_boundary(x_layer_out, h_post, x_res)

    def _eager_forward(
        self, x_layer_out: torch.Tensor, h_post: torch.Tensor, x_res: torch.Tensor
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        (batch, c_x), n_expand = x_layer_out.shape, h_post.shape[1]
        key = (batch, n_expand, c_x, x_layer_out.dtype, x_layer_out.device.index)
        inputs = tuple(t.contiguous() for t in (x_layer_out, h_post, x_res))
        self.kernel = self.kernel_for("mhc_post_kernel", inputs, key)
        return self.kernel(*inputs)
