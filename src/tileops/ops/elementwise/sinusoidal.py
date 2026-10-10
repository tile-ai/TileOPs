"""Sinusoidal positional encoding generative op."""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import SinusoidalFwdKernel
from tileops.kernels.elementwise.call_spec import SinusoidalCall, SinusoidalFwdInterface
from tileops.kernels.kernel_base import KernelInterface
from tileops.ops.elementwise._base import generated_on
from tileops.ops.op_base import Op


class SinusoidalFwdOp(Op):
    """Sinusoidal positional encoding from "Attention Is All You Need".

    Generates the full (seq_len, d_model) encoding tensor.

    Note:
        Eager-only. Unlike the other elementwise ops in this package,
        ``SinusoidalFwdOp`` is not registered as a ``torch.library.custom_op``,
        so ``torch.compile`` graph capture is not supported. The op has
        zero tensor inputs and constructs its output entirely from
        ``__init__`` parameters; no compile-time wrapping is needed.

    """

    kernel_types = {"sinusoidal": SinusoidalFwdKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "sinusoidal": SinusoidalFwdInterface
    }

    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        out_dtype: torch.dtype = torch.float32,
        device: "torch.device | str | None" = None,
        target: Target = None,
    ):
        """Build the op: the extents and dtype are its parameters.

        Args:
            seq_len: Sequence length.
            d_model: Model dimension.
            out_dtype: Dtype of the generated tensor.
            device: Where the tensor is produced, and so the device a target is detected
                from. ``None`` produces it on the current CUDA device, or wherever a named
                target decides.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from ``device``.
        """
        self.seq_len = seq_len
        self.d_model = d_model
        self.out_dtype = out_dtype
        self.device = device
        super().__init__(target=target)

    def forward(self) -> torch.Tensor:
        """Generate the tensor, in ``out_dtype`` whatever storage the kernel computes in."""
        call = SinusoidalCall(
            device=generated_on(self),
            seq_len=self.seq_len,
            d_model=self.d_model,
            dtype=self.out_dtype,
        )
        kernel = self.kernel_for("sinusoidal", call)
        out = kernel()
        return out if out.dtype == self.out_dtype else out.to(self.out_dtype)
