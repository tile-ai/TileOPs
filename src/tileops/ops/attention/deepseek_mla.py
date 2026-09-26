from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import MLADecodeWsKernel
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per ``(batch, heads, heads_kv, seqlen_kv, dim, pe_dim,
        dtype)``."""
        return call, lambda: self.kernel_map["mla_decode_kernel"](*call, tune=self.tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mla_decode_kernel": MLADecodeWsKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        q_pe_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        k_pe_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o.shape == q.shape``."""
        return {"o": tuple(q_shape)}

    def forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            q_pe: Input tensor, dtype ``same_as(q)``.
            k: Input tensor, dtype ``same_as(q)``.
            k_pe: Input tensor, dtype ``same_as(q)``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, H, D)``.
        """
        return self._wrapped(q, q_pe, k, k_pe, self._instance_key)

    def _eager_forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._validate_dtypes(q, q_pe, k, k_pe)
        self.dtype = q.dtype
        # The legacy roofline formula reads these off the op.
        self.batch, self.heads, self.dim = q.shape
        _, self.seqlen_kv, self.heads_kv, _ = k.shape
        self.pe_dim = q_pe.shape[2]
        inputs = (q, q_pe, k, k_pe)
        call = (
            self.batch,
            self.heads,
            self.heads_kv,
            self.seqlen_kv,
            self.dim,
            self.pe_dim,
            q.dtype,
        )
        return self.kernel_for("mla_decode_kernel", inputs, call)(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
