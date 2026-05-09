"""Sinusoidal positional encoding generative op."""

from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import SinusoidalFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY, _apply_fp8_post_cast


class SinusoidalFwdOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.
        dtype: Torch dtype.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "sinusoidal"
    _wrapped = None

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](seq_len, d_model, dtype)
        # Scalar tensor used as device/dtype carrier for torch.compile tracing
        self._device_carrier = torch.empty((), dtype=dtype, device="cuda")
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"sinusoidal": SinusoidalFwdKernel}

    def _eager_forward(self) -> torch.Tensor:
        out = self.kernel()
        result = out.reshape(self.seq_len, self.d_model)
        return _apply_fp8_post_cast(result, self.kernel)

    def forward(self) -> torch.Tensor:
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(
                self._device_carrier,
                self.seq_len, self.d_model,
                self._instance_key,
            )
        return self._eager_forward()
