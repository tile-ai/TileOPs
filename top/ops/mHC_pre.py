from typing import Dict, Optional

import torch

from top.kernels.mHC import mhc_pre_kernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["ManifoldConstrainedHyperConnectionPreOp"]


class ManifoldConstrainedHyperConnectionPreOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: str = 'float32',
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["mhc_pre_kernel"](
            batch, n_expand, c_x, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_pre_kernel": mhc_pre_kernel}

    def forward(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                        alpha_pre: float, alpha_post:float, alpha_res: float, sinkhorn_repeat: int) -> torch.Tensor:

        return self.kernel(phi, x, b,
                        alpha_pre, alpha_post, alpha_res, sinkhorn_repeat)
