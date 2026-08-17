from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    FlashAttnBwdPreprocessKernel,
    GQABwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAPrefillFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
    MHADecodeKernel,
    MHADecodePagedKernel,
    MHADecodePagedWsKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..compile_boundary import get_instance
from ..op_base import Op
from .gqa import GroupedQueryAttentionBwdOp, GroupedQueryAttentionFwdOp
from .selection import MHA_PAGED_DECODE_KEYS, AttentionCall

__all__ = [
    "MultiHeadAttentionBwdOp",
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp",
    "MultiHeadAttentionDecodeWithKVCacheFwdOp",
    "MultiHeadAttentionFwdOp",
]


class MultiHeadAttentionFwdOp(Op):
    """Layout: BSHD.

    MHA is the heads_kv == heads specialization of GQA, so route the
    maintained forward path through the GQA prefill dispatcher.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dispatch_kernel(kernel_map)
        self._gqa_op = GroupedQueryAttentionFwdOp(
            batch=batch,
            heads=heads,
            heads_kv=heads,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            kernel_map=self.forwarded_overrides(),
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_fwd_kernel": GQAPrefillFwdKernel,
            "gqa_prefill_causal_fwd_kernel": GQAPrefillFwdWsPersistentCausalKernel,
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
        }

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        return self._gqa_op._get_kernel(dtype)

    def kernel_delegates(self) -> tuple[GroupedQueryAttentionFwdOp, ...]:
        """Every kernel this op runs is built by the GQA prefill dispatcher."""
        return (self._gqa_op,)

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        return {"o": tuple(q_shape)}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run MHA forward."""
        return _mha_fwd(q, k, v, self._instance_key)

    def _eager_forward(self, q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor) -> torch.Tensor:
        self.dtype = q.dtype
        return self._gqa_op(q, k, v)


class MultiHeadAttentionBwdOp(Op):
    """Layout: BSHD.

    MHA backward is the ``heads_kv == heads`` specialization of GQA backward,
    matching the forward path's dispatch through GQA.
    """

    _LEGACY_KERNEL_MAP_KEYS = frozenset({
        "mha_bwd_preprocess_kernel",
        "mha_bwd_kernel",
        "mha_bwd_postprocess_kernel",
    })

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dispatch_kernel(self._gqa_kernel_map(kernel_map))
        self._gqa_op = GroupedQueryAttentionBwdOp(
            batch=batch,
            heads=heads,
            heads_kv=heads,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            kernel_map=self.forwarded_overrides(),
            tune=tune,
        )
        self.kernel_map = self._gqa_op.kernel_map

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel":
                GQABwdWgmmaPipelinedKernel,
        }

    def kernel_delegates(self) -> tuple[GroupedQueryAttentionBwdOp, ...]:
        """Every kernel this op runs is built by GQA backward."""
        return (self._gqa_op,)

    @staticmethod
    def _gqa_kernel_map(kernel_map: Optional[Dict[str, Kernel]]) -> Optional[Dict[str, Kernel]]:
        if kernel_map is None:
            return None
        legacy_keys = MultiHeadAttentionBwdOp._LEGACY_KERNEL_MAP_KEYS.intersection(kernel_map)
        if legacy_keys:
            keys = ", ".join(sorted(legacy_keys))
            raise ValueError(
                "MultiHeadAttentionBwdOp delegates to GroupedQueryAttentionBwdOp; "
                f"legacy MHA backward kernel_map keys are not compatible: {keys}. "
                "Use gqa_bwd_* keys with kernels that implement the GQA backward ABI.")
        return dict(kernel_map)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                do: torch.Tensor,
                lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._gqa_op(q, k, v, o, do, lse)


class MultiHeadAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "mha_decode_kernel",
            key=dtype,
            build=lambda: self.kernel_map["mha_decode_kernel"](
                self.batch, self.heads, self.seqlen_q, self.seqlen_kv,
                self.dim, False, dtype, tune=self.tune,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"mha_decode_kernel": MHADecodeKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k, v, real_seqlen_kv)


class MultiHeadAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged MHA decode with dynamic KV cache. Layout: Q [batch, seqlen_q, heads, dim] (BSHD);
    K, V physical cache [seqlen_kv, heads, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 seqlen_q: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 is_causal: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        call = self._attention_call(dtype)
        key = self.select_kernel_key(MHA_PAGED_DECODE_KEYS, call)

        def build() -> Kernel:
            return self.kernel_map[key](
                call.batch, call.heads, call.max_seqlen_q, call.seqlen_kv,
                call.dim, call.page_size, call.is_causal, dtype, tune=call.tune,
            )

        return self.get_or_build_kernel(key, key=dtype, build=build)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "mha_decode_paged_kernel": MHADecodePagedKernel,
            "mha_decode_paged_ws_kernel": MHADecodePagedWsKernel,
        }

    def _attention_call(self, dtype: torch.dtype) -> AttentionCall:
        """State what one paged decode call is, for selection to filter against.

        The element type arrives with the inputs rather than with the op, so one
        instance serves every dtype it is handed. Named with a leading underscore
        where the GQA siblings' equivalent is public: this round's provenance gate
        rejects any addition to a public Op surface under ``src/tileops/ops/``.
        """
        return AttentionCall(
            dtype=dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads,
            dim=self.dim,
            max_seqlen_q=self.seqlen_q,
            seqlen_kv=self.seqlen_kv,
            page_size=self.page_size,
            is_causal=self.is_causal,
            tune=self.tune,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k, v, real_seqlen_kv, block_table)


# torch.compile dispatch boundary (see src/tileops/ops/compile_boundary.py)


@torch.library.custom_op("top::mha_fwd", mutates_args=())
def _mha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             instance_key: str) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(q, k, v)


@_mha_fwd.register_fake
def _mha_fwd_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  instance_key: str) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(q.shape), tuple(k.shape), tuple(v.shape))
    return q.new_empty(shapes["o"])
