from typing import ClassVar, Mapping

import torch

from tileops._kernels.attention import (
    GQADecodePagedKernel,
    MHADecodePagedWSKernel,
)
from tileops._kernels.attention.call_spec import AttentionCall, MHAPagedDecodeFwdInterface
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = [
    "MHAPagedFwdOp",
]


class MHAPagedFwdOp(Op):
    """Paged Multi-Head Attention (MHA) decode with dynamic KV cache. Layout: ``Q`` $[batch \\times seqlen\\_q \\times heads \\times dim]$ (BSHD);
    k_cache, v_cache physical cache [seqlen_kv, heads, dim]; block_table [batch, num_pages]; cache_seqlens [batch].

    A causal call aligns the queries to the end of each request's cache: query ``i`` sees
    the keys up to position ``i + cache_seqlens - seqlen_q``. A query that sees no key
    outputs zeros. The op takes no ``sm_scale``; scores are scaled by ``dim ** -0.5``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mha_decode_paged_kernel": GQADecodePagedKernel,
        "mha_decode_paged_ws_kernel": MHADecodePagedWSKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mha_decode_paged": MHAPagedDecodeFwdInterface
    }

    def roofline_data_terms(self) -> "dict[str, int]":
        """The cached tokens this call's lengths name and the distinct pool rows they reach,
        which its flops and cache reads follow."""
        from tileops._perf.formulas import paged_decode_cache_rows

        call = self.last_call
        return {
            "kv_tokens": sum(call.metadata_values("cache_seqlens")),
            "cache_rows": paged_decode_cache_rows(call),
        }

    def __init__(
        self,
        page_size: int,
        is_causal: bool = False,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``False``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.page_size = page_size
        self.is_causal = is_causal
        super().__init__(target=target)

    def _attention_call(
        self, q: torch.Tensor, k_cache: torch.Tensor, block_table: torch.Tensor
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
            seqlen_kv=k_cache.shape[0],
            page_size=self.page_size,
            max_pages_per_req=block_table.shape[1],
            is_causal=self.is_causal,
            cache_dtype=k_cache.dtype,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k_cache: Input tensor, same dtype as ``q``.
            v_cache: Input tensor, same dtype as ``q``.
            block_table: Input tensor, dtype ``int32``.
            cache_seqlens: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, S_q, H, D)``.
        """
        inputs = (q, k_cache, v_cache, cache_seqlens, block_table)
        kernel = self.kernel_for(
            "mha_decode_paged", lambda: self._attention_call(q, k_cache, block_table)
        )
        return kernel(*inputs)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
