"""Sinusoidal positional encoding generative op."""

from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import SinusoidalFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op


class SinusoidalFwdOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Note:
        Eager-only. Unlike the other elementwise ops in this package,
        ``SinusoidalFwdOp`` is not registered as a ``torch.library.custom_op``,
        so ``torch.compile`` graph capture is not supported. The op has
        zero tensor inputs and constructs its output entirely from
        ``__init__`` parameters; no compile-time wrapping is needed.

    Args:
        seq_len: Sequence length.
        d_model: Model dimension.
        dtype: Torch dtype.
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
            the in-tree kernels, or ``None``. Nothing is probed: with no tensor input
            there is no device to detect, so the in-tree kernels serve unless a target
            is named (RFC §3.5).
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "sinusoidal"

    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        dtype: torch.dtype,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.seq_len = seq_len
        self.d_model = d_model
        self.dtype = dtype
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self):
        return {"sinusoidal": SinusoidalFwdKernel}

    def _infer_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"output": (self.seq_len, self.d_model)}

    def _validate_dtypes(self) -> None:
        return None

    @property
    def total_memory(self) -> int:
        return self.seq_len * self.d_model * self.dtype.itemsize

    def eval_roofline(self) -> tuple[int, int]:
        n_elem = self.seq_len * self.d_model
        return 6 * n_elem, self.total_memory

    def _build(self, dtype: torch.dtype):
        impl, ctor_dtype = self.kernel_map[self._op_name].specialize(dtype)
        return impl(self.seq_len, self.d_model, ctor_dtype)

    def forward(self) -> torch.Tensor:
        # The op promised ``self.dtype``; whichever storage the backend chose to
        # compute in is its own business and does not reach the caller.
        kernel = self.get_or_build_kernel(
            self._op_name,
            (),  # no tensor input: this op is in-tree only (RFC S3.5)
            key=self.dtype,
            build=lambda: self._build(self.dtype),
        )
        out = kernel().reshape(self.seq_len, self.d_model)
        return out if out.dtype == self.dtype else out.to(self.dtype)
