from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQADecodePagedKernel,
    MHADecodePagedWsKernel,
)
from tileops.kernels.attention.call_spec import AttentionCall
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = [
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
]


class MultiHeadAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged MHA decode with dynamic KV cache. Layout: ``Q`` $[batch \\times seqlen\\_q \\times heads \\times dim]$ (BSHD);
    K, V physical cache [seqlen_kv, heads, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].

    A causal call aligns the queries to the end of each request's cache: query ``i`` sees
    the keys up to position ``i + real_seqlen_kv - seqlen_q``. A query that sees no key
    outputs zeros.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mha_decode_paged_kernel": GQADecodePagedKernel,
        "mha_decode_paged_ws_kernel": MHADecodePagedWsKernel,
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The cached tokens this call's lengths name and the distinct pool rows they reach,
        which its flops and cache reads follow."""
        from tileops.perf.formulas import paged_decode_cache_rows

        call = self.last_call
        return {
            "kv_tokens": sum(call.values("real_seqlen_kv")),
            "cache_rows": paged_decode_cache_rows(call),
        }

    def __init__(
        self,
        page_size: int,
        is_causal: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``False``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.page_size = page_size
        self.is_causal = is_causal
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _attention_call(
        self, q: torch.Tensor, k: torch.Tensor, block_table: torch.Tensor
    ) -> AttentionCall:
        """State what one paged decode call is, for selection to filter against.

        Every extent and the element type arrive with the inputs, so one instance serves
        every shape and dtype it is handed.
        """
        batch, seqlen_q, heads, dim = q.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads,
            dim=dim,
            max_seqlen_q=seqlen_q,
            seqlen_kv=k.shape[0],
            page_size=self.page_size,
            max_pages_per_req=block_table.shape[1],
            is_causal=self.is_causal,
            cache_dtype=k.dtype,
            tune=self.tune,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            real_seqlen_kv: Input tensor, dtype ``int32``.
            block_table: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, S_q, H, D)``.
        """
        return self._call_boundary(q, k, v, real_seqlen_kv, block_table)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = (q, k, v, real_seqlen_kv, block_table)
        kernel = self.kernel_for(
            "mha_decode_paged", inputs, self._attention_call(q, k, block_table)
        )
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
