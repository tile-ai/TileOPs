"""ALiBi position-encoding generative op."""

from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import AlibiFwdKernel
from tileops.kernels.kernel_base import Entry, Kernel

from ..op_base import Op


class AlibiFwdOp(Op):
    """ALiBi position encoding: bias[h, i, j] = -slope_h * |i - j|.

    Generates the full (num_heads, seq_len, seq_len) bias tensor.

    Note:
        Eager-only. Unlike the other elementwise ops in this package,
        ``AlibiFwdOp`` is not registered as a ``torch.library.custom_op``,
        so ``torch.compile`` graph capture is not supported. The op has
        zero tensor inputs and constructs its output entirely from
        ``__init__`` parameters; no compile-time wrapping is needed.

    """

    _op_name = "alibi"

    def __init__(
        self,
        *,
        seq_len: int,
        num_heads: int,
        out_dtype: torch.dtype = torch.float32,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            out_dtype: Dtype of the generated tensor.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None``. Nothing is probed: with no tensor input
                there is no device to detect, so the in-tree kernels serve unless a target
                is named.
            kernel_map: Optional dispatch override mapping kernel keys to
                ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
        """
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.out_dtype = out_dtype
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self):
        return {"alibi": AlibiFwdKernel}

    def _infer_output_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"output": (self.num_heads, self.seq_len, self.seq_len)}

    def _validate_dtypes(self) -> None:
        return None

    @property
    def total_memory(self) -> int:
        return self.num_heads * self.seq_len * self.seq_len * self.out_dtype.itemsize

    def entry_for(self, role: str, call: torch.dtype) -> Entry:
        """One implementation, built per dtype; the extents are the op's."""
        return call, lambda: self._build(call)

    def _build(self, dtype: torch.dtype):
        impl, ctor_dtype = self.kernel_map[self._op_name].specialize(dtype)
        return impl(self.seq_len, self.num_heads, ctor_dtype)

    def forward(self) -> torch.Tensor:
        # The op promised ``self.out_dtype``; whichever storage the backend chose to
        # compute in is its own business and does not reach the caller.
        """Run the op on the inputs the manifest declares.

        Returns:
            ``output``, as the manifest declares.
        """
        # No tensor input, so no device to detect: in-tree only.
        kernel = self.kernel_for(self._op_name, (), self.out_dtype)
        out = kernel().reshape(self.num_heads, self.seq_len, self.seq_len)
        return out if out.dtype == self.out_dtype else out.to(self.out_dtype)
