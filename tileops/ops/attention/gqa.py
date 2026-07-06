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
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAFwdWsPersistentKernel,
    GQAPrefillFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeAppendKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQASlidingWindowFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
    GQASlidingWindowVarlenFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import is_h200, is_hopper

from ..op_base import Op
from ..rope import _base_freqs

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionFwdOp",
    "GroupedQueryAttentionPrefillFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
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


def _validate_attention_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Expected dtype torch.float16 or torch.bfloat16, got {dtype}")


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


def _select_gqa_prefill_fwd_kernel_cls(
    dim: int,
    is_causal: bool,
    dtype: torch.dtype,
    sm_scale: float,
    softcap: float,
    *,
    hopper: bool,
) -> Type[Kernel]:
    del sm_scale, softcap
    if hopper and is_causal and dim == 128 and dtype in (torch.float16, torch.bfloat16):
        return GQAPrefillFwdWsPersistentCausalKernel
    return GQAPrefillFwdKernel


def _select_gqa_prefill_kernel_key(
    *,
    backend: str,
    is_fp8: bool,
    uses_sliding_window: bool,
    is_uniform: bool,
) -> str:
    """Select the concrete canonical prefill kernel key without instantiating kernels."""
    if is_fp8:
        if backend not in ("auto", "fp8"):
            raise ValueError("FP8 prefill requires backend='auto' or backend='fp8'.")
        return "gqa_prefill_fp8_tensor_core_fwd_kernel"
    if uses_sliding_window:
        if backend not in ("auto", "sliding_window"):
            raise ValueError(
                "sliding-window prefill requires backend='auto' or backend='sliding_window'.")
        return "gqa_sliding_window_varlen_fwd"
    if is_uniform and backend in ("auto", "dense"):
        return "gqa_prefill_fwd_kernel"
    if backend == "dense":
        raise ValueError("backend='dense' requires uniform packed cu_seqlens.")
    if backend not in ("auto", "varlen"):
        raise ValueError("non-FP8 prefill requires backend='auto', 'dense', or 'varlen'.")
    return "gqa_prefill_varlen_fwd_kernel"


def _select_gqa_paged_prefill_kernel_keys(
    *,
    cache_dtype: torch.dtype,
    attention_dtype: torch.dtype,
    fuse_rope: bool,
) -> tuple[str, ...]:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fuse_rope:
        if cache_dtype != attention_dtype:
            raise ValueError("fuse_rope is not supported with FP8 paged KV cache yet")
        return (
            "gqa_prefill_paged_with_kv_cache_rope_append_kernel",
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
        )
    if fp8_dtype is not None and cache_dtype == fp8_dtype:
        return ("gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",)
    return ("gqa_prefill_paged_with_kv_cache_fwd_kernel",)


def _select_gqa_prefill_varlen_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillVarlenFwdKernel


def _select_gqa_prefill_paged_with_kv_cache_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillPagedWithKVCacheFwdKernel


def _select_gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel_cls() -> Type[Kernel]:
    # Selector hook for future FP8 paged-cache kernel variants.
    return GQAPrefillPagedWithFP8KVCacheFwdKernel


def _select_gqa_prefill_paged_with_kv_cache_rope_fwd_kernel_cls() -> Type[Kernel]:
    return GQAPrefillPagedWithKVCacheRopeFwdKernel


def _select_gqa_prefill_paged_with_kv_cache_rope_append_kernel_cls() -> Type[Kernel]:
    return GQAPrefillPagedWithKVCacheRopeAppendKernel


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


def _score_softcap(softcap: Optional[float]) -> float:
    if softcap is None:
        return 0.0
    if softcap < 0:
        raise ValueError("softcap must be non-negative")
    return softcap


def _rope_rotary_dim(dim: int, rotary_dim: Optional[int]) -> int:
    rotary_dim = dim if rotary_dim is None else rotary_dim
    if rotary_dim <= 0:
        raise ValueError("rotary_dim must be positive")
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    if rotary_dim > dim:
        raise ValueError("rotary_dim must not exceed dim")
    return rotary_dim


def _attention_output(result: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if isinstance(result, torch.Tensor):
        return result
    output, _ = result
    return output


class GroupedQueryAttentionFwdOp(Op):
    """Compatibility square GQA forward wrapper. Public layout: BSHD."""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
                 dtype: torch.dtype = torch.float16,
                 sm_scale: Optional[float] = None,
                 softcap: Optional[float] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        self.dispatch_kernel(kernel_map)
        self._prefill_op = GroupedQueryAttentionPrefillFwdOp(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_seqlen_q=seq_len,
            max_seqlen_kv=seq_len,
            dtype=dtype,
            is_causal=is_causal,
            sm_scale=sm_scale,
            softcap=softcap,
            backend="dense",
            kernel_map=self.kernel_map,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        dense_kernel_cls = _select_gqa_prefill_fwd_kernel_cls(
            self.dim,
            self.is_causal,
            self.dtype,
            hopper=is_hopper(),
            sm_scale=self.sm_scale,
            softcap=self.softcap,
        )
        return {"gqa_prefill_fwd_kernel": dense_kernel_cls}

    @property
    def kernel(self) -> Kernel:
        return self._prefill_op._get_dense_kernel()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        expected_q = (self.batch, self.seq_len, self.heads, self.dim)
        expected_kv = (self.batch, self.seq_len, self.heads_kv, self.dim)
        if tuple(q.shape) != expected_q:
            raise ValueError(f"q must have shape {expected_q}, got {tuple(q.shape)}")
        if tuple(k.shape) != expected_kv:
            raise ValueError(f"k must have shape {expected_kv}, got {tuple(k.shape)}")
        if tuple(v.shape) != expected_kv:
            raise ValueError(f"v must have shape {expected_kv}, got {tuple(v.shape)}")
        if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
            raise ValueError(f"q/k/v dtype must match op dtype {self.dtype}.")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        kernel = (
            self._prefill_op._get_square_dense_kernel()
            if self._prefill_op._uses_square_dense_fast_path()
            else self._prefill_op._get_dense_kernel()
        )
        return _attention_output(kernel(q, k, v))


class GroupedQueryAttentionPrefillFwdOp(Op):
    """Canonical packed GQA prefill. Layout: THD.

    Dense and square prefill are represented with uniform ``cu_seqlens``. Ragged
    prefill uses the same fixed public tensor list. Scale tensors are required
    for manifest stability; non-FP8 kernels ignore them.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dtype: torch.dtype = torch.float16,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        backend: str = "auto",
        validate_uniform_cu_seqlens: bool = True,
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
        if is_causal and max_seqlen_q > max_seqlen_kv:
            raise ValueError("causal prefill requires max_seqlen_q <= max_seqlen_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}")
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}")
        if backend not in ("auto", "dense", "varlen", "fp8", "sliding_window"):
            raise ValueError(
                "backend must be one of 'auto', 'dense', 'varlen', 'fp8', or 'sliding_window'"
            )
        if backend == "auto" and not validate_uniform_cu_seqlens:
            raise ValueError("backend='auto' requires validate_uniform_cu_seqlens=True.")
        _validate_attention_dtype(dtype)

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dtype = dtype
        self.is_causal = is_causal
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.backend = backend
        self.validate_uniform_cu_seqlens = validate_uniform_cu_seqlens
        self.tune = tune
        self._dense_kernel = None
        self._square_dense_kernel = None
        self._varlen_kernel = None
        self._sliding_window_varlen_kernel = None
        self._fp8_kernel = None
        self._roofline_kwargs = None
        self._uniform_cu_cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        dense_kernel_cls = _select_gqa_prefill_fwd_kernel_cls(
            self.dim,
            self.is_causal,
            self.dtype,
            self.sm_scale,
            self.softcap,
            hopper=is_hopper(),
        )
        sliding_kernel_cls = (
            GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
            if is_hopper()
            else GQASlidingWindowVarlenFwdKernel
        )
        return {
            "gqa_prefill_fwd_kernel": dense_kernel_cls,
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
            "gqa_prefill_varlen_fwd_kernel": _select_gqa_prefill_varlen_fwd_kernel_cls(),
            "gqa_sliding_window_varlen_fwd": sliding_kernel_cls,
            "gqa_prefill_fp8_tensor_core_fwd_kernel":
                GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_kv_shape: tuple[int, ...],
        q_scale_shape: tuple[int, ...],
        k_scale_shape: tuple[int, ...],
        v_scale_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        return {"o": tuple(q_shape)}

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        is_fp8 = fp8_dtype is not None and q.dtype == fp8_dtype
        if is_fp8:
            if k.dtype != fp8_dtype or v.dtype != fp8_dtype:
                raise ValueError("FP8 prefill requires q/k/v to all be torch.float8_e4m3fn.")
        else:
            if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
                raise ValueError(f"q/k/v dtype must match op dtype {self.dtype}.")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_kv.dtype != torch.int32:
            raise ValueError("cu_seqlens_q/cu_seqlens_kv must be torch.int32.")
        if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
            raise ValueError("q_scale/k_scale/v_scale must be torch.float32.")

    def _validate_common_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None:
        for tensor, name in (
            (q, "q"),
            (k, "k"),
            (v, "v"),
            (cu_seqlens_q, "cu_seqlens_q"),
            (cu_seqlens_kv, "cu_seqlens_kv"),
            (q_scale, "q_scale"),
            (k_scale, "k_scale"),
            (v_scale, "v_scale"),
        ):
            if tensor.device.type != "cuda":
                raise ValueError(f"{name} must be on a cuda device, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if q.ndim != 3 or tuple(q.shape[1:]) != (self.heads, self.dim):
            raise ValueError(
                f"q must have shape [T, {self.heads}, {self.dim}], got {tuple(q.shape)}")
        if k.ndim != 3 or tuple(k.shape[1:]) != (self.heads_kv, self.dim):
            raise ValueError(
                f"k must have shape [T, {self.heads_kv}, {self.dim}], got {tuple(k.shape)}")
        if v.ndim != 3 or tuple(v.shape[1:]) != (self.heads_kv, self.dim):
            raise ValueError(
                f"v must have shape [T, {self.heads_kv}, {self.dim}], got {tuple(v.shape)}")
        if v.shape[0] != k.shape[0]:
            raise ValueError(f"v.shape[0] ({v.shape[0]}) must equal k.shape[0] ({k.shape[0]})")
        expected_cu_shape = (self.batch + 1,)
        if tuple(cu_seqlens_q.shape) != expected_cu_shape:
            raise ValueError(
                f"cu_seqlens_q must have shape {expected_cu_shape}, got {tuple(cu_seqlens_q.shape)}"
            )
        if tuple(cu_seqlens_kv.shape) != expected_cu_shape:
            raise ValueError(
                f"cu_seqlens_kv must have shape {expected_cu_shape}, got {tuple(cu_seqlens_kv.shape)}"
            )
        expected_scale_shape = (self.batch, self.heads_kv)
        for tensor, name in ((q_scale, "q_scale"), (k_scale, "k_scale"), (v_scale, "v_scale")):
            if tuple(tensor.shape) != expected_scale_shape:
                raise ValueError(
                    f"{name} must have shape {expected_scale_shape}, got {tuple(tensor.shape)}"
                )

    def _uniform_cu_seqlens(self, cu_seqlens: torch.Tensor, seq_len: int) -> bool:
        cache_key = (seq_len, cu_seqlens.device, cu_seqlens.dtype)
        expected = self._uniform_cu_cache.get(cache_key)
        if expected is None:
            expected = torch.arange(
                self.batch + 1,
                device=cu_seqlens.device,
                dtype=cu_seqlens.dtype,
            ) * seq_len
            self._uniform_cu_cache[cache_key] = expected
        return bool(torch.equal(cu_seqlens, expected))

    def _uses_sliding_window(self) -> bool:
        return self.window_size_left != -1 or self.window_size_right != -1

    def _is_fp8_tensor(self, tensor: torch.Tensor) -> bool:
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        return fp8_dtype is not None and tensor.dtype == fp8_dtype

    def _get_dense_kernel(self) -> Kernel:
        if self._dense_kernel is None:
            self._dense_kernel = self.kernel_map["gqa_prefill_fwd_kernel"](
                self.batch,
                self.heads,
                self.heads_kv,
                self.max_seqlen_q,
                self.max_seqlen_kv,
                self.dim,
                self.is_causal,
                self.dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=self.tune,
            )
        return self._dense_kernel

    def _uses_square_dense_fast_path(self) -> bool:
        if self.dtype not in (torch.float16, torch.bfloat16):
            return False
        if not is_h200() or self.dim != 128:
            return False
        if self.heads % self.heads_kv != 0 or self.max_seqlen_q % _WS_BLOCK_M != 0:
            return False
        m_blocks = math.ceil(self.max_seqlen_q / _WS_BLOCK_M)
        if m_blocks % 2 != 0:
            return False
        return (
            self.is_causal
            and self.max_seqlen_q == self.max_seqlen_kv
            and _gqa_ws_causal_total_work_items(
                self.batch, self.heads, self.heads_kv, self.max_seqlen_q
            ) >= _H200_SMS
        )

    def _get_square_dense_kernel(self) -> Kernel:
        if self._square_dense_kernel is None:
            self._square_dense_kernel = self.kernel_map["gqa_prefill_square_fwd_kernel"](
                self.batch,
                self.heads,
                self.heads_kv,
                self.max_seqlen_q,
                self.dim,
                self.is_causal,
                self.dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=self.tune,
            )
        return self._square_dense_kernel

    def _get_varlen_kernel(self) -> Kernel:
        if self._varlen_kernel is None:
            self._varlen_kernel = self.kernel_map["gqa_prefill_varlen_fwd_kernel"](
                batch=self.batch,
                heads=self.heads,
                heads_kv=self.heads_kv,
                dim=self.dim,
                is_causal=self.is_causal,
                dtype=self.dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=self.tune,
            )
        return self._varlen_kernel

    def _get_sliding_window_varlen_kernel(self) -> Kernel:
        if self._sliding_window_varlen_kernel is None:
            self._sliding_window_varlen_kernel = self.kernel_map[
                "gqa_sliding_window_varlen_fwd"](
                    batch=self.batch,
                    heads=self.heads,
                    heads_kv=self.heads_kv,
                    dim=self.dim,
                    is_causal=self.is_causal,
                    window_size_left=self.window_size_left,
                    window_size_right=self.window_size_right,
                    dtype=self.dtype,
                    accum_dtype=torch.float32,
                    tune=self.tune,
                )
        return self._sliding_window_varlen_kernel

    def _get_fp8_kernel(self) -> Kernel:
        if self._fp8_kernel is None:
            self._fp8_kernel = self.kernel_map["gqa_prefill_fp8_tensor_core_fwd_kernel"](
                self.batch,
                self.heads,
                self.heads_kv,
                self.max_seqlen_q,
                self.dim,
                self.dtype,
                tune=self.tune,
            )
        return self._fp8_kernel

    def _record_roofline(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
        self._roofline_kwargs = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch": self.batch,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "is_causal": self.is_causal,
            "dtype": self.dtype,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_dtypes(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale)
        self._validate_common_shapes(q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale,
                                     v_scale)
        if self.backend == "auto" or (
            self.backend in ("dense", "fp8") and self.validate_uniform_cu_seqlens
        ):
            q_uniform = self._uniform_cu_seqlens(cu_seqlens_q, self.max_seqlen_q)
            kv_uniform = self._uniform_cu_seqlens(cu_seqlens_kv, self.max_seqlen_kv)
            is_uniform = q_uniform and kv_uniform
        else:
            is_uniform = True

        kernel_key = _select_gqa_prefill_kernel_key(
            backend=self.backend,
            is_fp8=self._is_fp8_tensor(q),
            uses_sliding_window=self._uses_sliding_window(),
            is_uniform=is_uniform,
        )

        if kernel_key == "gqa_prefill_fp8_tensor_core_fwd_kernel":
            if self.is_causal:
                raise ValueError("FP8 Tensor Core prefill currently supports non-causal prefill only.")
            if self._uses_sliding_window():
                raise ValueError("FP8 Tensor Core prefill does not support sliding-window dispatch.")
            if self.max_seqlen_q != self.max_seqlen_kv:
                raise ValueError("FP8 Tensor Core prefill requires max_seqlen_q == max_seqlen_kv.")
            if not is_uniform:
                raise ValueError("FP8 Tensor Core prefill requires uniform packed cu_seqlens.")
            q_bshd = q.view(self.batch, self.max_seqlen_q, self.heads, self.dim)
            k_bshd = k.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim)
            v_bshd = v.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim)
            out = _attention_output(self._get_fp8_kernel()(q_bshd, k_bshd, v_bshd, q_scale,
                                                            k_scale, v_scale))
            self._record_roofline(q, k, cu_seqlens_q, cu_seqlens_kv)
            return out.reshape(q.shape)

        if kernel_key == "gqa_sliding_window_varlen_fwd":
            output, _ = self._get_sliding_window_varlen_kernel()(
                q, k, v, cu_seqlens_q, cu_seqlens_kv, self.max_seqlen_q)
            self._record_roofline(q, k, cu_seqlens_q, cu_seqlens_kv)
            return output

        if kernel_key == "gqa_prefill_fwd_kernel":
            q_bshd = q.view(self.batch, self.max_seqlen_q, self.heads, self.dim)
            k_bshd = k.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim)
            v_bshd = v.view(self.batch, self.max_seqlen_kv, self.heads_kv, self.dim)
            kernel = (
                self._get_square_dense_kernel()
                if self._uses_square_dense_fast_path()
                else self._get_dense_kernel()
            )
            out = _attention_output(kernel(q_bshd, k_bshd, v_bshd))
            self._record_roofline(q, k, cu_seqlens_q, cu_seqlens_kv)
            return out.reshape(q.shape)

        output, _ = self._get_varlen_kernel()(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, self.max_seqlen_q, self.max_seqlen_kv)
        self._record_roofline(q, k, cu_seqlens_q, cu_seqlens_kv)
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call")
        from tileops.perf.formulas import gqa_prefill_varlen_fwd_roofline

        kwargs = dict(self._roofline_kwargs)
        kwargs["q_lens"] = GroupedQueryAttentionPrefillVarlenFwdOp._lengths_from_cu_seqlens(
            kwargs.pop("cu_seqlens_q"))
        kwargs["kv_lens"] = GroupedQueryAttentionPrefillVarlenFwdOp._lengths_from_cu_seqlens(
            kwargs.pop("cu_seqlens_kv"))
        return gqa_prefill_varlen_fwd_roofline(**kwargs)



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
        softcap: Optional[float] = None,
        validate_inputs: bool = False,
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
        _validate_attention_dtype(dtype)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        self.validate_inputs = validate_inputs
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
            softcap=self.softcap,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_prefill_varlen_fwd_kernel": _select_gqa_prefill_varlen_fwd_kernel_cls()}

    @staticmethod
    def _lengths_from_cu_seqlens(cu_seqlens: torch.Tensor) -> list[int]:
        values = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        return [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
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

        if v.shape[0] != k.shape[0]:
            raise ValueError(f"v.shape[0] ({v.shape[0]}) must equal k.shape[0] ({k.shape[0]})")
        if not self.validate_inputs:
            return

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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_forward_inputs(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        output, _ = self.kernel(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, self.max_seqlen_q, self.max_seqlen_kv)
        self._roofline_kwargs = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch": self.batch,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "is_causal": self.is_causal,
            "dtype": self.dtype,
        }
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call")
        from tileops.perf.formulas import gqa_prefill_varlen_fwd_roofline

        kwargs = dict(self._roofline_kwargs)
        kwargs["q_lens"] = self._lengths_from_cu_seqlens(kwargs.pop("cu_seqlens_q"))
        kwargs["kv_lens"] = self._lengths_from_cu_seqlens(kwargs.pop("cu_seqlens_kv"))
        return gqa_prefill_varlen_fwd_roofline(**kwargs)



class GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(Op):
    """Packed GQA prefill with paged KV cache append. Layout: THD.

    The current chunk is packed by request. ``cache_seqlens`` stores each
    request's logical KV length before append. ``block_table`` maps logical
    page ids to physical pages in ``k_pages`` / ``v_pages``.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        max_pages_per_req: int,
        page_size: int,
        dim: int,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
        cache_dtype: torch.dtype | str | None = None,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        fuse_rope: bool = False,
        rope_base: float = 10000.0,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        if fuse_rope:
            rotary_dim = _rope_rotary_dim(dim, rotary_dim)
            if max_position is None:
                raise ValueError("max_position is required when fuse_rope=True")
            if max_position <= 0:
                raise ValueError("max_position must be positive")
        elif rotary_dim is not None:
            raise ValueError("rotary_dim requires fuse_rope=True")
        if batch <= 0:
            raise ValueError("batch must be positive")
        if max_pages_per_req <= 0:
            raise ValueError("max_pages_per_req must be positive")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        if page_size & (page_size - 1) != 0:
            raise ValueError("page_size must be a power of two")
        _validate_attention_dtype(dtype)
        if cache_dtype is None:
            cache_dtype = dtype
        elif isinstance(cache_dtype, str):
            if cache_dtype == "float8_e4m3fn":
                fp8_dtype = getattr(torch, "float8_e4m3fn", None)
                if fp8_dtype is None:
                    raise ValueError(
                        "torch.float8_e4m3fn is not supported on this PyTorch version")
                cache_dtype = fp8_dtype
            else:
                candidate = getattr(torch, cache_dtype, None)
                if not isinstance(candidate, torch.dtype):
                    raise ValueError(f"Unknown cache_dtype {cache_dtype}")
                cache_dtype = candidate
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if cache_dtype != dtype and cache_dtype != fp8_dtype:
            raise ValueError(
                "cache_dtype must be either same as dtype or torch.float8_e4m3fn, "
                f"got {cache_dtype}")
        if fuse_rope and cache_dtype == fp8_dtype:
            raise ValueError("fuse_rope is not supported with FP8 paged KV cache yet")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.groups = heads // heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.max_cache_len = max_pages_per_req * page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.cache_dtype = cache_dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        self.fuse_rope = fuse_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._rope_cos_cache: Optional[torch.Tensor] = None
        self._rope_sin_cache: Optional[torch.Tensor] = None
        self._rope_cache_device: Optional[torch.device] = None

        self.dispatch_kernel(kernel_map)
        self.append_kernel: Optional[Kernel] = None
        if self.fuse_rope:
            self.append_kernel = self.kernel_map[
                "gqa_prefill_paged_with_kv_cache_rope_append_kernel"](
                    batch=batch,
                    heads_kv=heads_kv,
                    max_pages_per_req=max_pages_per_req,
                    page_size=page_size,
                    dim=dim,
                    max_position=self.max_position,
                    rotary_dim=self.rotary_dim,
                    dtype=dtype,
                    tune=tune,
                )
            self.kernel = self.kernel_map["gqa_prefill_paged_with_kv_cache_rope_fwd_kernel"](
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                max_position=self.max_position,
                rotary_dim=self.rotary_dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=tune,
            )
        elif self.cache_dtype == fp8_dtype:
            self.kernel = self.kernel_map["gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel"](
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=tune,
            )
        else:
            self.kernel = self.kernel_map["gqa_prefill_paged_with_kv_cache_fwd_kernel"](
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                max_pages_per_req=max_pages_per_req,
                page_size=page_size,
                dim=dim,
                is_causal=is_causal,
                dtype=dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=tune,
            )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        keys = _select_gqa_paged_prefill_kernel_keys(
            cache_dtype=self.cache_dtype,
            attention_dtype=self.dtype,
            fuse_rope=self.fuse_rope,
        )
        if keys == (
            "gqa_prefill_paged_with_kv_cache_rope_append_kernel",
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
        ):
            return {
                "gqa_prefill_paged_with_kv_cache_rope_append_kernel":
                    _select_gqa_prefill_paged_with_kv_cache_rope_append_kernel_cls(),
                "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel":
                    _select_gqa_prefill_paged_with_kv_cache_rope_fwd_kernel_cls()
            }
        if keys == ("gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",):
            return {
                "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel":
                    _select_gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel_cls()
            }
        return {
            "gqa_prefill_paged_with_kv_cache_fwd_kernel":
                _select_gqa_prefill_paged_with_kv_cache_fwd_kernel_cls()
        }

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
    ) -> None:
        tensors = {
            "q": q,
            "k_new": k_new,
            "v_new": v_new,
            "k_pages": k_pages,
            "v_pages": v_pages,
            "k_scale": k_scale,
            "v_scale": v_scale,
            "cu_seqlens_q": cu_seqlens_q,
            "cache_seqlens": cache_seqlens,
            "block_table": block_table,
        }
        for name, tensor in tensors.items():
            if tensor.device.type != "cuda":
                raise ValueError(f"{name} must be on a cuda device, got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        expected_q_shape_tail = (self.heads, self.dim)
        expected_kv_shape_tail = (self.heads_kv, self.dim)
        if q.ndim != 3 or tuple(q.shape[1:]) != expected_q_shape_tail:
            raise ValueError(
                f"q must have shape [total_q, {self.heads}, {self.dim}], got {q.shape}")
        if k_new.ndim != 3 or tuple(k_new.shape[1:]) != expected_kv_shape_tail:
            raise ValueError(
                f"k_new must have shape [total_q, {self.heads_kv}, {self.dim}], got "
                f"{k_new.shape}")
        if v_new.shape != k_new.shape:
            raise ValueError(
                f"v_new must have the same shape as k_new, got {v_new.shape} and "
                f"{k_new.shape}")
        if k_new.shape[0] != q.shape[0]:
            raise ValueError(
                f"k_new.shape[0] ({k_new.shape[0]}) must equal q.shape[0] ({q.shape[0]})")
        if k_pages.ndim != 3 or tuple(k_pages.shape[1:]) != expected_kv_shape_tail:
            raise ValueError(
                f"k_pages must have shape [physical_tokens, {self.heads_kv}, {self.dim}], "
                f"got {k_pages.shape}")
        if v_pages.shape != k_pages.shape:
            raise ValueError(
                f"v_pages must have the same shape as k_pages, got {v_pages.shape} and "
                f"{k_pages.shape}")
        if k_pages.shape[0] % self.page_size != 0:
            raise ValueError("k_pages physical token dimension must be divisible by page_size")
        if k_scale.shape != (1,) or v_scale.shape != (1,):
            raise ValueError(
                f"k_scale and v_scale must have shape (1,), got {k_scale.shape} and "
                f"{v_scale.shape}")
        if cu_seqlens_q.shape != (self.batch + 1,):
            raise ValueError(
                f"cu_seqlens_q shape must be ({self.batch + 1},), got "
                f"{tuple(cu_seqlens_q.shape)}")
        if cache_seqlens.shape != (self.batch,):
            raise ValueError(
                f"cache_seqlens shape must be ({self.batch},), got "
                f"{tuple(cache_seqlens.shape)}")
        if block_table.shape != (self.batch, self.max_pages_per_req):
            raise ValueError(
                f"block_table shape must be ({self.batch}, {self.max_pages_per_req}), "
                f"got {tuple(block_table.shape)}")

        for name, tensor in [("q", q), ("k_new", k_new), ("v_new", v_new)]:
            if tensor.dtype != self.dtype:
                raise ValueError(f"Expected {name}.dtype {self.dtype}, got {tensor.dtype}")
        for name, tensor in [("k_pages", k_pages), ("v_pages", v_pages)]:
            if tensor.dtype != self.cache_dtype:
                raise ValueError(
                    f"Expected {name}.dtype {self.cache_dtype}, got {tensor.dtype}")
        for name, tensor in [("k_scale", k_scale), ("v_scale", v_scale)]:
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
            if self.cache_dtype == getattr(torch, "float8_e4m3fn", None) and not torch.all(
                torch.isfinite(tensor) & (tensor > 0)
            ).item():
                raise ValueError(f"{name} must contain finite positive values")
        for name, tensor in [("cu_seqlens_q", cu_seqlens_q),
                             ("cache_seqlens", cache_seqlens),
                             ("block_table", block_table)]:
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")

        if int(cu_seqlens_q[0].item()) != 0:
            raise ValueError("cu_seqlens_q[0] must be 0")
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if torch.any(q_lens < 0).item():
            raise ValueError("cu_seqlens_q must be non-decreasing")
        total_q = int(cu_seqlens_q[-1].item())
        if total_q != q.shape[0]:
            raise ValueError(f"cu_seqlens_q[-1] ({total_q}) must equal q.shape[0] ({q.shape[0]})")
        actual_max_q = int(q_lens.max().item())
        if max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})")

        min_cache_len = int(cache_seqlens.min().item())
        max_total_len = int((cache_seqlens + q_lens).max().item())
        if min_cache_len < 0:
            raise ValueError("cache_seqlens must be non-negative")
        if max_total_len > self.max_cache_len:
            raise ValueError(
                "cache_seqlens + q_len exceeds paged KV capacity: "
                f"max total length {max_total_len}, capacity {self.max_cache_len}")
        if self.fuse_rope and max_total_len > self.max_position:
            raise ValueError(
                "cache_seqlens + q_len exceeds RoPE max_position: "
                f"max total length {max_total_len}, max_position {self.max_position}")

        num_pages = k_pages.shape[0] // self.page_size
        min_page = int(block_table.min().item())
        max_page = int(block_table.max().item())
        if min_page < 0:
            raise ValueError("block_table must contain non-negative physical page ids")
        if max_page >= num_pages:
            raise ValueError(
                f"block_table references page {max_page}, but only {num_pages} pages exist")

    def _get_rope_cos_sin(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_position is None:
            raise ValueError("max_position is required when fuse_rope=True")
        if self._rope_cos_cache is None or self._rope_cache_device != device:
            self._rope_cos_cache, self._rope_sin_cache = _base_freqs(
                self.rotary_dim,
                self.max_position,
                base=self.rope_base,
                dtype=self.dtype,
                device=device,
            )
            self._rope_cache_device = device
        return self._rope_cos_cache, self._rope_sin_cache

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        self._validate_forward_inputs(
            q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_seqlens_q, cache_seqlens,
            block_table, max_seqlen_q)
        if self.cache_dtype == getattr(torch, "float8_e4m3fn", None):
            return _attention_output(
                self.kernel(q, k_new, v_new, k_pages, v_pages, k_scale, v_scale, cu_seqlens_q,
                            cache_seqlens, block_table, max_seqlen_q))
        if self.fuse_rope:
            cos, sin = self._get_rope_cos_sin(q.device)
            self.append_kernel(
                k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table,
                max_seqlen_q, cos, sin)
            return _attention_output(
                self.kernel(q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens,
                            block_table, max_seqlen_q, cos, sin))
        return _attention_output(
            self.kernel(q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens,
                        block_table, max_seqlen_q))

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time.")

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time.")



class GroupedQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool = True,
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
                 sm_scale: Optional[float] = None,
                 softcap: Optional[float] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        _validate_attention_dtype(dtype)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_kernel"](
            batch,
            heads,
            heads_kv,
            seqlen_kv,
            dim,
            self.dtype,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            tune=tune,
        )

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
                 sm_scale: Optional[float] = None,
                 softcap: Optional[float] = None,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        _validate_attention_dtype(dtype)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_paged_kernel"](
            batch,
            heads,
            heads_kv,
            seqlen_kv,
            dim,
            page_size,
            self.dtype,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            tune=tune,
        )

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
        is_causal: bool = True,
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
        is_causal: bool = True,
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
