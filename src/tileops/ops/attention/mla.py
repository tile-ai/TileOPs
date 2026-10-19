from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    MLADecodeMMAKernel,
    MLADecodeWSKernel,
    MLAPagedFwdKernel,
    MLAPagedSeesawKernel,
    MLAVarlenPrefillFwdKernel,
    MLAVarlenPrefillWSFwdKernel,
)
from tileops.kernels.attention.call_spec import (
    MLADecodeCall,
    MLADecodeFwdInterface,
    MLAPagedCall,
    MLAPagedFwdInterface,
    MLAVarlenCall,
    MLAVarlenFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = [
    "MLADecodeWithKVCacheFwdOp",
    "MLAPagedFwdOp",
    "MLAVarlenFwdOp",
]


class MLADecodeWithKVCacheFwdOp(Op):
    """Multi-Head Latent Attention (MLA) decode against a per-request KV cache. Layout: BSHD."""

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_decode_kernel": MLADecodeWSKernel,
        "mla_decode_mma_kernel": MLADecodeMMAKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_decode_kernel": MLADecodeFwdInterface
    }

    def __init__(
        self,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self, q: torch.Tensor, q_pe: torch.Tensor, k: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            q_pe: Input tensor, same dtype as ``q``.
            k: Input tensor, same dtype as ``q``.
            k_pe: Input tensor, same dtype as ``q``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, H, D)``.
        """
        batch, heads, dim = q.shape
        _, seqlen_kv, heads_kv, _ = k.shape
        inputs = (q, q_pe, k, k_pe)
        call = MLADecodeCall(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seqlen_kv=seqlen_kv,
            dim=dim,
            pe_dim=q_pe.shape[2],
            dtype=q.dtype,
            device=q.device,
        )
        return self.kernel_for("mla_decode_kernel", call)(*inputs)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class MLAPagedFwdOp(Op):
    """Multi-Head Latent Attention (MLA) decode over a paged latent cache, absorbed form.

    The cache holds one row per token, ``kv_c`` followed by ``k_pe``, shared by every query
    head: the key is the whole row and the value its first ``kv_lora_rank`` columns. The
    ``S_q`` query tokens of a request are its last ``S_q`` cache positions (one for decode,
    more for multi-token prediction drafts). ``lse`` is the float32 natural-log log-sum-exp
    of each row's scaled scores, for a caller merging partial results. With an FP8 cache
    the scores and the output are multiplied by ``kv_scale``; the cached values are used
    exactly. A cache row past a request's ``cache_seqlens`` never reaches a result,
    whatever it holds.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_paged": MLAPagedFwdKernel,
        "mla_paged_seesaw": MLAPagedSeesawKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"mla_paged": MLAPagedFwdInterface}

    def __init__(
        self,
        kv_lora_rank: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and dtypes come from each call.

        Args:
            kv_lora_rank: Width of the value: the cache row's first columns.
            is_causal: Whether a query token sees only the keys at or before its position.
            sm_scale: Score scale, or ``None`` for ``DK ** -0.5``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.kv_lora_rank = kv_lora_rank
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        super().__init__(target=target)

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        kv_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend each request's query tokens to its cache.

        Args:
            q: ``(B, S_q, H, DK)`` queries, latent part then rope part, float16 or bfloat16.
            kv_cache: ``(NP, PS, DK)`` latent cache in the query dtype or float8_e4m3fn.
            block_table: ``int32`` ``(B, W)``; key ``n`` of request ``b`` is row
                ``n % PS`` of page ``block_table[b, n // PS]``.
            cache_seqlens: ``int32`` ``(B,)``, the keys each request holds.
            kv_scale: ``float32`` ``(1,)`` dequantization scale, required exactly when
                the cache is float8_e4m3fn.

        Returns:
            ``o`` ``(B, S_q, H, kv_lora_rank)`` in the query dtype and ``lse``
            ``(B, S_q, H)`` float32.
        """
        q = q.contiguous()
        block_table = block_table.contiguous()
        cache_seqlens = cache_seqlens.contiguous()
        kv_scale = None if kv_scale is None else kv_scale.contiguous()
        call = MLAPagedCall(
            s_q=q.shape[1],
            heads=q.shape[2],
            dim=q.shape[3],
            dim_v=self.kv_lora_rank,
            page_size=kv_cache.shape[1],
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            dtype=q.dtype,
            cache_dtype=kv_cache.dtype,
            device=q.device,
        )
        return self.kernel_for("mla_paged", call)(q, kv_cache, block_table, cache_seqlens, kv_scale)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])


class MLAVarlenFwdOp(Op):
    """Multi-Head Latent Attention (MLA) prefill over packed requests, after the latent is
    decompressed. Layout: THD.

    The key of head ``h`` is ``k_nope[:, h]`` followed by ``k_pe``, which holds one row per
    token and is shared by every head. Queries and keys are the same tokens, so one
    ``cu_seqlens`` describes both and the causal mask sits on the diagonal. ``lse`` is
    returned in float32 so a caller merging chunked-context partials can combine them.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_varlen_fwd": MLAVarlenPrefillFwdKernel,
        "mla_varlen_fwd_ws": MLAVarlenPrefillWSFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_varlen_fwd": MLAVarlenFwdInterface
    }

    def __init__(
        self,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Whether a query row reads only keys at or before its own position.
            sm_scale: Score scale, or ``None`` for ``(DN + PE) ** -0.5``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        super().__init__(target=target)

    def varlen_call(self, inputs: tuple) -> MLAVarlenCall:
        """State what one contiguous call is, for selection to filter against."""
        q, k_nope, k_pe, _v, cu_seqlens = inputs
        return MLAVarlenCall(
            batch=cu_seqlens.shape[0] - 1,
            heads=q.shape[1],
            dim_nope=k_nope.shape[2],
            dim_pe=k_pe.shape[1],
            dim_v=_v.shape[2],
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            dtype=q.dtype,
            device=q.device,
        )

    def forward(
        self,
        q: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k_nope: Input tensor, same dtype as ``q``.
            k_pe: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            cu_seqlens: Packed request offsets, dtype ``int32``.

        Returns:
            ``o`` and ``lse``, as the manifest declares. Shape rules:
            ``o.shape == (T_q, H, DV)`` and ``lse.shape == (T_q, H)``.
        """
        inputs = tuple(tensor.contiguous() for tensor in (q, k_nope, k_pe, v, cu_seqlens))
        return self.kernel_for("mla_varlen_fwd", self.varlen_call(inputs))(*inputs)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
