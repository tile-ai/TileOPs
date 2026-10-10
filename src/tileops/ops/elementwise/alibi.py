"""ALiBi position-encoding generative op."""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import AlibiFwdKernel
from tileops.kernels.elementwise.call_spec import AlibiCall, AlibiFwdInterface
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.elementwise._base import generated_on
from tileops.ops.op_base import Op


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
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"alibi": AlibiFwdInterface}

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
                ``Kernel`` subclasses. Falls back to ``kernel_types``.
            tune: Whether to autotune.
        """
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.out_dtype = out_dtype
        self.device = device
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def forward(self) -> torch.Tensor:
        """Generate the tensor, in ``out_dtype`` whatever storage the kernel computes in."""
        call = AlibiCall(
            device=generated_on(self),
            seq_len=self.seq_len,
            num_heads=self.num_heads,
            dtype=self.out_dtype,
        )
        kernel = self.kernel_for("alibi", call)
        out = kernel()
        return out if out.dtype == self.out_dtype else out.to(self.out_dtype)
