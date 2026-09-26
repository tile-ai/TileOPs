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

    kernel_types = {"alibi": AlibiFwdKernel}

    def __init__(
        self,
        *,
        seq_len: int,
        num_heads: int,
        out_dtype: torch.dtype = torch.float32,
        device: "torch.device | str | None" = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op: the extents and dtype are its parameters.

        Args:
            seq_len: Sequence length.
            num_heads: Number of attention heads.
            out_dtype: Dtype of the generated tensor.
            device: Where the tensor is produced, and so the device a target is detected
                from. ``None`` produces it on the current CUDA device, or wherever a named
                target decides.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from ``device``.
            kernel_map: Optional dispatch override mapping kernel keys to
                ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
            tune: Whether to autotune.
        """
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.out_dtype = out_dtype
        self.device = device
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per dtype and device; the extents are the op's."""
        return call, lambda: self._build(*call)

    def _build(self, dtype: torch.dtype, device_index: "int | None" = None):
        impl, ctor_dtype = self.kernel_map["alibi"].specialize(dtype)
        return impl(
            self.seq_len, self.num_heads, ctor_dtype, tune=self.tune, device_index=device_index
        )

    def forward(self) -> torch.Tensor:
        """Generate the tensor, in ``out_dtype`` whatever storage the kernel computes in."""
        device = self._declared_device()
        index = None if device is None else device.index
        kernel = self.kernel_for("alibi", (), (self.out_dtype, index))
        out = kernel().reshape(self.num_heads, self.seq_len, self.seq_len)
        return out if out.dtype == self.out_dtype else out.to(self.out_dtype)
