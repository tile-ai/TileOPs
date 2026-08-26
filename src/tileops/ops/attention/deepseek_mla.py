from typing import Dict, Optional

import torch

from tileops.kernels.attention import MLADecodeWsKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op, tensor_core_roof

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        pe_dim: int,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            pe_dim: Manifest ``params.pe_dim``, ``int``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.pe_dim = pe_dim

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "mla_decode_kernel",
            inputs,
            key=dtype,
            build=lambda: self.kernel_map["mla_decode_kernel"](
                self.batch,
                self.heads,
                self.heads_kv,
                self.seqlen_kv,
                self.dim,
                self.pe_dim,
                dtype,
                tune=self.tune,
            ),
        )

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
        self._validate_dtypes(q, q_pe, k, k_pe)
        self.dtype = q.dtype
        return self._get_kernel((q, q_pe, k, k_pe), q.dtype)(q, q_pe, k, k_pe)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
