import math
from typing import Dict, Optional, Type

import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQABwdKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodeKernel,
    GQADecodePagedKernel,
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
    GQAPrefillFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQAPrefillWithKVCacheFwdKernel,
    GQASlidingWindowFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
    GQASlidingWindowVarlenFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import is_h200, is_hopper

from ..op_base import Op

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionFwdOp",
    "GroupedQueryAttentionPrefillFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionPrefillWithKVCacheFwdOp",
    "GroupedQueryAttentionSlidingWindowFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
]

_WS_BLOCK_M = 128
_H200_SMS = 132


def _gqa_ws_noncausal_total_work_items(batch: int, heads: int, heads_kv: int, seq_len: int) -> int:
    groups = heads // heads_kv
    m_blocks = math.ceil(seq_len / _WS_BLOCK_M)
    return batch * heads_kv * m_blocks * groups


def _gqa_ws_causal_total_work_items(batch: int, heads: int, heads_kv: int, seq_len: int) -> int:
    groups = heads // heads_kv
    m_blocks = math.ceil(seq_len / _WS_BLOCK_M)
    return batch * heads_kv * (m_blocks // 2) * groups


def _supports_gqa_ws_noncausal(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    dtype: torch.dtype,
    *,
    h200: bool,
    num_sms: int = _H200_SMS,
) -> bool:
    if not h200 or dtype != torch.float16 or dim != 128:
        return False
    if heads % heads_kv != 0 or seq_len % _WS_BLOCK_M != 0:
        return False
    return _gqa_ws_noncausal_total_work_items(batch, heads, heads_kv, seq_len) >= num_sms


def _supports_gqa_ws_causal(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    dtype: torch.dtype,
    *,
    h200: bool,
    num_sms: int = _H200_SMS,
) -> bool:
    if not h200 or dtype != torch.float16 or dim != 128:
        return False
    if heads % heads_kv != 0 or seq_len % _WS_BLOCK_M != 0:
        return False
    m_blocks = math.ceil(seq_len / _WS_BLOCK_M)
    if m_blocks % 2 != 0:
        return False
    return _gqa_ws_causal_total_work_items(batch, heads, heads_kv, seq_len) >= num_sms


def _select_gqa_fwd_kernel_cls(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: torch.dtype,
    *,
    hopper: bool,
    h200: bool,
) -> Type[Kernel]:
    if not hopper:
        return GQAFwdKernel
    if is_causal:
        if _supports_gqa_ws_causal(batch, heads, heads_kv, seq_len, dim, dtype, h200=h200):
            return GQAFwdWsPersistentCausalKernel
        return GQAFwdWgmmaPipelinedKernel
    if _supports_gqa_ws_noncausal(batch, heads, heads_kv, seq_len, dim, dtype, h200=h200):
        return GQAFwdWsPersistentKernel
    return GQAFwdWgmmaPipelinedKernel


def _select_gqa_prefill_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillFwdKernel


def _select_gqa_prefill_varlen_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillVarlenFwdKernel


def _select_gqa_prefill_with_kv_cache_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillWithKVCacheFwdKernel


def _validate_gqa_dims(heads: int, heads_kv: int, dim: int) -> None:
    if heads <= 0:
        raise ValueError("heads must be positive")
    if heads_kv <= 0:
        raise ValueError("heads_kv must be positive")
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    if dim <= 0:
        raise ValueError("dim must be positive")


def _attention_scale(dim: int, sm_scale: Optional[float]) -> float:
    return dim**-0.5 if sm_scale is None else sm_scale


def _attention_output(result: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    output, _ = result
    return output


class GroupedQueryAttentionFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_fwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel_cls = _select_gqa_fwd_kernel_cls(
            self.batch,
            self.heads,
            self.heads_kv,
            self.seq_len,
            self.dim,
            self.is_causal,
            self.dtype,
            hopper=is_hopper(),
            h200=is_h200(),
        )
        return {"gqa_fwd_kernel": kernel_cls}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v)


class GroupedQueryAttentionPrefillFwdOp(Op):
    """Dense GQA prefill. Layout: BSHD.

    Supports ``seq_len_q != seq_len_kv``. Causal prefill uses bottom-right
    alignment: key position ``j`` is visible to query position ``i`` iff
    ``j <= i + (seq_len_kv - seq_len_q)``.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len_q: int,
                 seq_len_kv: int,
                 dim: int,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 sm_scale: Optional[float] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        if is_causal and seq_len_q > seq_len_kv:
            raise ValueError("causal prefill requires seq_len_q <= seq_len_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_prefill_fwd_kernel"](
            batch, heads, heads_kv, seq_len_q, seq_len_kv, dim, is_causal, self.dtype,
            sm_scale=self.sm_scale, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_prefill_fwd_kernel": _select_gqa_prefill_fwd_kernel_cls()}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return _attention_output(self.kernel(q, k, v))


class GroupedQueryAttentionPrefillVarlenFwdOp(Op):
    """Packed variable-length GQA prefill. Layout: THD.

    ``cu_seqlens_q`` and ``cu_seqlens_kv`` describe packed per-request ranges.
    Causal prefill uses bottom-right alignment for each request independently:
    key position ``j`` is visible to query position ``i`` iff
    ``j <= i + (kv_len - q_len)``.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        sm_scale: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        if batch <= 0:
            raise ValueError("batch must be positive")
        if max_seqlen_q <= 0:
            raise ValueError("max_seqlen_q must be positive")
        if max_seqlen_kv <= 0:
            raise ValueError("max_seqlen_kv must be positive")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self._roofline_kwargs = None

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_prefill_varlen_fwd_kernel"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=is_causal,
            dtype=dtype,
            sm_scale=self.sm_scale,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_prefill_varlen_fwd_kernel": _select_gqa_prefill_varlen_fwd_kernel_cls()}

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        tensors = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
        }
        for name, tensor in tensors.items():
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        expected_tail_shapes = {
            "q": (self.heads, self.dim),
            "k": (self.heads_kv, self.dim),
            "v": (self.heads_kv, self.dim),
        }
        for name, expected_tail in expected_tail_shapes.items():
            tensor = tensors[name]
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != expected_tail:
                raise ValueError(
                    f"Expected {name} shape [T, {expected_tail[0]}, {expected_tail[1]}], "
                    f"got {tuple(tensor.shape)}")
            if tensor.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {tensor.dtype}")

        for name in ("cu_seqlens_q", "cu_seqlens_kv"):
            tensor = tensors[name]
            expected_shape = (self.batch + 1,)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.int32:
                raise ValueError(f"Expected {name}.dtype torch.int32, got {tensor.dtype}")

        cu_q = [int(x) for x in cu_seqlens_q.detach().cpu().tolist()]
        cu_kv = [int(x) for x in cu_seqlens_kv.detach().cpu().tolist()]
        if cu_q[0] != 0:
            raise ValueError(f"cu_seqlens_q[0] must be 0, got {cu_q[0]}")
        if cu_kv[0] != 0:
            raise ValueError(f"cu_seqlens_kv[0] must be 0, got {cu_kv[0]}")
        if cu_q[-1] != q.shape[0]:
            raise ValueError(
                f"cu_seqlens_q[-1] ({cu_q[-1]}) must equal q.shape[0] ({q.shape[0]})")
        if cu_kv[-1] != k.shape[0]:
            raise ValueError(
                f"cu_seqlens_kv[-1] ({cu_kv[-1]}) must equal k.shape[0] ({k.shape[0]})")
        if v.shape[0] != k.shape[0]:
            raise ValueError(f"v.shape[0] ({v.shape[0]}) must equal k.shape[0] ({k.shape[0]})")
        if any(cu_q[i + 1] < cu_q[i] for i in range(self.batch)):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if any(cu_kv[i + 1] < cu_kv[i] for i in range(self.batch)):
            raise ValueError("cu_seqlens_kv must be non-decreasing")

        q_lens = []
        kv_lens = []
        for idx in range(self.batch):
            q_len = cu_q[idx + 1] - cu_q[idx]
            kv_len = cu_kv[idx + 1] - cu_kv[idx]
            q_lens.append(q_len)
            kv_lens.append(kv_len)
            if q_len <= 0:
                raise ValueError("all q sequence lengths must be positive")
            if kv_len <= 0:
                raise ValueError("all kv sequence lengths must be positive")
            if self.is_causal and q_len > kv_len:
                raise ValueError("causal varlen prefill requires every q_len <= kv_len")
        actual_max_q = max(q_lens)
        actual_max_kv = max(kv_lens)
        if self.max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({self.max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})")
        if self.max_seqlen_kv < actual_max_kv:
            raise ValueError(
                f"max_seqlen_kv ({self.max_seqlen_kv}) must be >= actual max KV "
                f"sequence length ({actual_max_kv})")
        return q_lens, kv_lens

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        q_lens, kv_lens = self._validate_forward_inputs(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        output, _ = self.kernel(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, self.max_seqlen_q, self.max_seqlen_kv)
        self._roofline_kwargs = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch": self.batch,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            "is_causal": self.is_causal,
            "dtype": self.dtype,
        }
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call")
        from tileops.perf.formulas import gqa_prefill_varlen_fwd_roofline

        result = gqa_prefill_varlen_fwd_roofline(**self._roofline_kwargs)
        return result["flops"], result["bytes"]


class GroupedQueryAttentionPrefillWithKVCacheFwdOp(Op):
    """Dense GQA prefill with contiguous KV cache append. Layout: BSHD.

    ``cache_seqlens`` stores the per-batch KV length before append. The fused
    kernel computes attention over old cache plus current ``k_new/v_new`` and
    appends the current chunk into ``k_cache/v_cache`` in-place.
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len_new: int,
                 seqlen_kv: int,
                 dim: int,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 sm_scale: Optional[float] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        if seq_len_new > seqlen_kv:
            raise ValueError("seq_len_new must not exceed seqlen_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_new = seq_len_new
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_prefill_with_kv_cache_fwd_kernel"](
            batch, heads, heads_kv, seq_len_new, seqlen_kv, dim, is_causal, self.dtype,
            sm_scale=self.sm_scale, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_with_kv_cache_fwd_kernel":
                _select_gqa_prefill_with_kv_cache_fwd_kernel_cls()
        }

    def _validate_forward_inputs(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                                 k_cache: torch.Tensor, v_cache: torch.Tensor,
                                 cache_seqlens: torch.Tensor) -> None:
        tensors = {
            "q": q,
            "k_new": k_new,
            "v_new": v_new,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "cache_seqlens": cache_seqlens,
        }
        for name, tensor in tensors.items():
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        expected_shapes = {
            "q": (self.batch, self.seq_len_new, self.heads, self.dim),
            "k_new": (self.batch, self.seq_len_new, self.heads_kv, self.dim),
            "v_new": (self.batch, self.seq_len_new, self.heads_kv, self.dim),
            "k_cache": (self.batch, self.seqlen_kv, self.heads_kv, self.dim),
            "v_cache": (self.batch, self.seqlen_kv, self.heads_kv, self.dim),
            "cache_seqlens": (self.batch,),
        }
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(tensors[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {actual_shape}")

        for name, tensor in tensors.items():
            if name == "cache_seqlens":
                continue
            if tensor.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {tensor.dtype}")
        if cache_seqlens.dtype != torch.int32:
            raise ValueError(f"Expected cache_seqlens.dtype torch.int32, got {cache_seqlens.dtype}")

        min_cache_len = int(cache_seqlens.min().item())
        max_cache_len = int(cache_seqlens.max().item())
        if min_cache_len < 0:
            raise ValueError("cache_seqlens must be non-negative")
        if max_cache_len + self.seq_len_new > self.seqlen_kv:
            raise ValueError(
                "cache_seqlens + seq_len_new exceeds KV cache capacity: "
                f"max cache_seqlen {max_cache_len}, seq_len_new {self.seq_len_new}, "
                f"seqlen_kv {self.seqlen_kv}")

    def forward(self, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor,
                cache_seqlens: torch.Tensor) -> torch.Tensor:
        self._validate_forward_inputs(q, k_new, v_new, k_cache, v_cache, cache_seqlens)
        return _attention_output(self.kernel(q, k_new, v_new, k_cache, v_cache, cache_seqlens))


class GroupedQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel":
                GQABwdWgmmaPipelinedKernel if is_hopper() else GQABwdKernel,
            "gqa_bwd_postprocess_kernel":
                FlashAttnBwdPostprocessKernel if not is_hopper() else None,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                do: torch.Tensor,
                lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        self.kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(self.dtype) if is_hopper() else self.post_kernel(dq)
        dk, dv = dk.to(self.dtype), dv.to(self.dtype)
        return dq, dk, dv


class GroupedQueryAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_kernel"](
            batch, heads, heads_kv, seqlen_kv, dim, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_decode_kernel": GQADecodeKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)

        return self.kernel(q, k, v, real_seqlen_kv)


class GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged GQA decode with dynamic KV cache. Layout: Q [batch, heads, dim] (BHD);
    K, V physical cache [seqlen_kv, heads_kv, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_paged_kernel"](
            batch, heads, heads_kv, seqlen_kv, dim, page_size, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_decode_paged_kernel": GQADecodePagedKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v, real_seqlen_kv, block_table)


class GroupedQueryAttentionSlidingWindowFwdOp(Op):
    """Fixed-length GQA forward with sliding window attention.

    Token at q_pos attends to k_pos when ALL applicable conditions hold:
      - k_pos <= q_pos                          (is_causal=True)
      - k_pos >= q_pos - window_size_left       (window_size_left >= 0)
      - k_pos <= q_pos + window_size_right      (window_size_right >= 0)

    Use window_size_left=-1 / window_size_right=-1 for no restriction.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        seq_len: Sequence length (same for Q, K, V).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        kernel_map: Optional override for hardware-specific kernel dispatch.
        tune: Whether to run autotuning on kernel instantiation.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}")
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_sliding_window_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel = GQASlidingWindowFwdWgmmaPipelinedKernel if is_hopper() else GQASlidingWindowFwdKernel
        return {"gqa_sliding_window_fwd": kernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Run fixed-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [batch, seq_len, heads, dim].
            k: Key tensor, shape [batch, seq_len, heads_kv, dim].
            v: Value tensor, shape [batch, seq_len, heads_kv, dim].

        Returns:
            Output tensor, shape [batch, seq_len, heads, dim].
        """
        for t, name in [(q, 'q'), (k, 'k'), (v, 'v')]:
            if t.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {t.device}")
            if t.dtype != self.dtype:
                raise ValueError(
                    f"{name} dtype {t.dtype} does not match op dtype {self.dtype}")
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        if q.shape != (self.batch, self.seq_len, self.heads, self.dim):
            raise ValueError(
                f"q shape {q.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads}, {self.dim})")
        if k.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"k shape {k.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})")
        if v.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"v shape {v.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})")

        output, _ = self.kernel.forward(q, k, v)
        return output

    @property
    def total_flops(self) -> int:
        """Approximate FLOPs for QK^T and PV GEMMs."""
        S = self.seq_len
        wl = self.window_size_left
        wr = self.window_size_right
        total_attended = 0
        for q in range(S):
            hi = q if self.is_causal else (min(S - 1, q + wr) if wr >= 0 else S - 1)
            lo = max(0, q - wl) if wl >= 0 else 0
            total_attended += hi - lo + 1
        return 4 * self.batch * self.heads * total_attended * self.dim

    @property
    def total_memory(self) -> int:
        """Approximate bytes accessed: read Q/K/V, write O."""
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.batch * self.seq_len * (self.heads + self.heads_kv) * self.dim * elem


class GroupedQueryAttentionSlidingWindowVarlenFwdOp(Op):
    """Variable-length GQA forward with sliding window attention.

    Inputs are packed (no padding); per-sample boundaries are given via
    cu_seqlens arrays.  seqlen_q and seqlen_k may differ per sample:

      offset = seqlen_k - seqlen_q  (per sample, FA3 bottom-right convention)

    A token at local q_pos attends to local k_pos when ALL conditions hold:
      k_pos <= q_pos + offset                      (is_causal=True)
      k_pos >= q_pos + offset - window_size_left   (window_size_left >= 0)
      k_pos <= q_pos + offset + window_size_right  (window_size_right >= 0)

    Args:
        batch: Number of sequences in the batch.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        accum_dtype: Accumulator data type for intermediate computations.
        kernel_map: Optional override for hardware-specific kernel dispatch.
        tune: Whether to run autotuning on kernel instantiation.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}")
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_sliding_window_varlen_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
            accum_dtype=accum_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel = (GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
                  if is_hopper() else GQASlidingWindowVarlenFwdKernel)
        return {"gqa_sliding_window_varlen_fwd": kernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Run variable-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [total_q, heads, dim].
            k: Key tensor, shape [total_k, heads_kv, dim].
            v: Value tensor, shape [total_k, heads_kv, dim].
            cu_seqlens_q: Cumulative Q lengths, shape [batch+1], dtype int32.
            cu_seqlens_k: Cumulative K lengths, shape [batch+1], dtype int32.
            max_seqlen_q: Maximum Q sequence length across the batch.

        Returns:
            Output tensor, shape [total_q, heads, dim].
        """
        for t, name in [(q, 'q'), (k, 'k'), (v, 'v')]:
            if t.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {t.device}")
            if t.dtype != self.dtype:
                raise ValueError(
                    f"{name} dtype {t.dtype} does not match op dtype {self.dtype}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if q.ndim != 3 or q.shape[1] != self.heads or q.shape[2] != self.dim:
            raise ValueError(
                f"q shape {q.shape} incompatible with heads={self.heads}, "
                f"dim={self.dim}")
        if k.ndim != 3 or k.shape[1] != self.heads_kv or k.shape[2] != self.dim:
            raise ValueError(
                f"k shape {k.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}")
        if v.ndim != 3 or v.shape[1] != self.heads_kv or v.shape[2] != self.dim:
            raise ValueError(
                f"v shape {v.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}")
        if cu_seqlens_q.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_q.shape[0] ({cu_seqlens_q.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})")
        if cu_seqlens_k.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_k.shape[0] ({cu_seqlens_k.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})")
        for cu, name in [(cu_seqlens_q, 'cu_seqlens_q'),
                         (cu_seqlens_k, 'cu_seqlens_k')]:
            if cu.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {cu.device}")
            if cu.dtype != torch.int32:
                raise ValueError(
                    f"{name} must have dtype int32, got {cu.dtype}")
            if not cu.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if cu_seqlens_q[0].item() != 0:
            raise ValueError(
                f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}")
        if cu_seqlens_k[0].item() != 0:
            raise ValueError(
                f"cu_seqlens_k[0] must be 0, got {cu_seqlens_k[0].item()}")
        if not torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if not torch.all(cu_seqlens_k[1:] >= cu_seqlens_k[:-1]):
            raise ValueError("cu_seqlens_k must be non-decreasing")
        if cu_seqlens_q[-1].item() > q.shape[0]:
            raise ValueError(
                f"cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()}) exceeds "
                f"q.shape[0] ({q.shape[0]})")
        if cu_seqlens_k[-1].item() > k.shape[0]:
            raise ValueError(
                f"cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()}) exceeds "
                f"k.shape[0] ({k.shape[0]})")
        actual_max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})")

        output, _ = self.kernel.forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
        return output

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time.")

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time.")
