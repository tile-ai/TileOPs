from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mhc import MHCPostKernel

from ..op import Op

__all__ = ["MHCPostOp"]


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
