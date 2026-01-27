from typing import Dict, Optional

import torch

from top.kernels.mHC import mhc_post_kernel
from top.kernels.kernel import Kernel

from .op import Op

__all__ = ["ManifoldConstrainedHyperConnectionPostOp"]


class ManifoldConstrainedHyperConnectionPostOp(Op):
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
        self.kernel = self.kernel_map["mhc_post_kernel"](
            batch, n_expand, c_x, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mhc_post_kernel": mhc_post_kernel}

    def forward(self, x_layer_out: torch.Tensor, h_post: torch.Tensor, x_res: torch.Tensor) -> torch.Tensor:

        return self.kernel(x_layer_out, h_post, x_res)
