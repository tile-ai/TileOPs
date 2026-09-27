from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import MLADecodeWsKernel
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["MultiHeadLatentAttentionDecodeWithKVCacheFwdOp"]


class MultiHeadLatentAttentionDecodeWithKVCacheFwdOp(Op):
    """Multi-Head Latent Attention (MLA) decode against a per-request KV cache. Layout: BSHD.

    The in-tree kernel serves a single KV head (``H_kv == 1``) and refuses others.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"mla_decode_kernel": MLADecodeWsKernel}

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per ``(batch, heads, heads_kv, seqlen_kv, dim, pe_dim,
        dtype)``."""
        return call, lambda: self.kernel_map["mla_decode_kernel"](*call, tune=self.tune)

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
        return self._call_boundary(q, q_pe, k, k_pe)

    def _eager_forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, heads, dim = q.shape
        _, seqlen_kv, heads_kv, _ = k.shape
        inputs = (q, q_pe, k, k_pe)
        call = (batch, heads, heads_kv, seqlen_kv, dim, q_pe.shape[2], q.dtype)
        return self.kernel_for("mla_decode_kernel", inputs, call)(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
