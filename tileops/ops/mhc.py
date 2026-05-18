from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mhc import MHCPostKernel, MHCPreKernel

from .op_base import Op

__all__ = ["MHCPostOp", "MHCPreOp"]


class MHCPreOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: torch.dtype = torch.float32,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mhc_pre_kernel"](batch, n_expand, c_x, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_pre_kernel": MHCPreKernel}

    def forward(self,
                phi: torch.Tensor,
                x: torch.Tensor,
                b: torch.Tensor,
                alpha_pre: float,
                alpha_post: float,
                alpha_res: float,
                sinkhorn_repeat: int,
                sinkhorn_eps: float = 0.02) -> torch.Tensor:

        return self.kernel(phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat,
                           sinkhorn_eps)


class MHCPostOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: torch.dtype = torch.float32,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mhc_post_kernel"](
            batch, n_expand, c_x, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_post_kernel": MHCPostKernel}

    def forward(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                x_res: torch.Tensor) -> torch.Tensor:

        return self.kernel(x_layer_out, h_post, x_res)
