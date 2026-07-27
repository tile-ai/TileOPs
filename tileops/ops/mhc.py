import math
from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mhc import MHCPostKernel, MHCPreKernel

from .op_base import Op

__all__ = ["MHCPostOp", "MHCPreOp"]


class MHCPreOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = None
        self.n_expand = None
        self.c_x = None
        self.dtype = None
        self.weights_dtype = torch.float32
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_pre_kernel": MHCPreKernel}

    @staticmethod
    def _n_expand_from_phi_dim(phi_dim: int) -> int:
        n_expand = int(math.isqrt(phi_dim + 1) - 1)
        if n_expand <= 0 or n_expand * n_expand + 2 * n_expand != phi_dim:
            raise ValueError(
                "phi.shape[1] must equal n_expand * n_expand + 2 * n_expand"
            )
        return n_expand

    def _get_kernel(
        self,
        batch: int,
        n_expand: int,
        c_x: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, n_expand, c_x, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["mhc_pre_kernel"](
                batch, n_expand, c_x, dtype, tune=self.tune,
            )
        return self._kernel_cache[key]

    def forward(self,
                phi: torch.Tensor,
                x: torch.Tensor,
                b: torch.Tensor,
                alpha_pre: float,
                alpha_post: float,
                alpha_res: float,
                sinkhorn_repeat: int,
                sinkhorn_eps: float = 0.02) -> torch.Tensor:

        if phi.ndim != 2 or x.ndim != 2 or b.ndim != 1:
            raise ValueError("MHCPreOp expects phi/x/b shapes [D, P], [B, D], [P]")
        batch, x_dim = x.shape
        if phi.shape[0] != x_dim:
            raise ValueError(f"phi.shape[0] must match x.shape[1]={x_dim}, got {phi.shape[0]}")
        n_expand = self._n_expand_from_phi_dim(phi.shape[1])
        if b.shape[0] != phi.shape[1]:
            raise ValueError(f"b.shape[0] must match phi.shape[1]={phi.shape[1]}, got {b.shape[0]}")
        if x_dim % n_expand != 0:
            raise ValueError(f"x.shape[1]={x_dim} must be divisible by n_expand={n_expand}")
        c_x = x_dim // n_expand
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = x.dtype
        self.kernel = self._get_kernel(batch, n_expand, c_x, x.dtype, x.device.index)
        return self.kernel(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat,
                           sinkhorn_eps)


class MHCPostOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = None
        self.n_expand = None
        self.c_x = None
        self.dtype = None
        self.weights_dtype = torch.float32
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_post_kernel": MHCPostKernel}

    def _get_kernel(
        self,
        batch: int,
        n_expand: int,
        c_x: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, n_expand, c_x, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["mhc_post_kernel"](
                batch, n_expand, c_x, dtype, tune=self.tune,
            )
        return self._kernel_cache[key]

    def forward(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                x_res: torch.Tensor) -> torch.Tensor:

        if x_layer_out.ndim != 2 or h_post.ndim != 2 or x_res.ndim != 2:
            raise ValueError("MHCPostOp expects x_layer_out/h_post/x_res to be 2D tensors")
        batch, c_x = x_layer_out.shape
        if h_post.shape[0] != batch or x_res.shape[0] != batch:
            raise ValueError("MHCPostOp inputs must have matching batch dimensions")
        n_expand = h_post.shape[1]
        if x_res.shape[1] != n_expand * c_x:
            raise ValueError(
                f"x_res.shape[1] must equal n_expand * c_x={n_expand * c_x}, got {x_res.shape[1]}"
            )
        if x_res.dtype != x_layer_out.dtype:
            raise ValueError(
                f"x_res.dtype must match x_layer_out.dtype ({x_layer_out.dtype}), got {x_res.dtype}"
            )
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = x_layer_out.dtype
        self.kernel = self._get_kernel(batch, n_expand, c_x, x_layer_out.dtype, x_layer_out.device.index)
        return self.kernel(x_layer_out, h_post, x_res)
