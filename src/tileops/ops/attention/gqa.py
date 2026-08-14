from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    FlashAttnBwdPreprocessKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodePagedBs1Kernel,
    GQADecodePagedKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAPrefillFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
    GQAPrefillPagedNativeFP8TensorCoreFwdKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillVarlenFP8TensorCoreFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from .selection import (
    DECODE_KEYS,
    DENSE_PREFILL_KEYS,
    PAGED_DECODE_KEYS,
    PAGED_PREFILL_KEYS,
    VARLEN_PREFILL_KEYS,
    AttentionCall,
    fp8_dtype,
)

_ROPE_LAYOUTS = frozenset(("neox", "interleaved"))

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillDenseFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
]


def _validate_attention_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Expected dtype torch.float16 or torch.bfloat16, got {dtype}")


def _paged_cache_dtype(cache_dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    """Validate a paged KV cache element type; ``None`` follows the attention dtype."""
    if cache_dtype is None:
        return None
    if cache_dtype != fp8_dtype():
        _validate_attention_dtype(cache_dtype)
    return cache_dtype


def _validate_positive(**values: int) -> None:
    """Raise for the first named value that is not positive; the name appears
    in the message, so pass the caller's own parameter name."""
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _validate_gqa_dims(heads: int, heads_kv: int, dim: int) -> None:
    _validate_positive(heads=heads, heads_kv=heads_kv)
    if heads % heads_kv != 0:
        raise ValueError("heads must be divisible by heads_kv")
    _validate_positive(dim=dim)


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
    _validate_positive(rotary_dim=rotary_dim)
    if rotary_dim % 2 != 0:
        raise ValueError("rotary_dim must be even")
    if rotary_dim > dim:
        raise ValueError("rotary_dim must not exceed dim")
    return rotary_dim


def _validate_rope_config(
    dim: int,
    fuse_rope: bool,
    rotary_dim: Optional[int],
    rope_layout: str,
) -> Optional[int]:
    """Validate static RoPE semantics shared by every prefill topology."""
    if rope_layout not in _ROPE_LAYOUTS:
        raise ValueError(f"rope_layout must be one of {sorted(_ROPE_LAYOUTS)}, got {rope_layout!r}")
    if fuse_rope:
        return _rope_rotary_dim(dim, rotary_dim)
    if rotary_dim is not None:
        raise ValueError("rotary_dim requires fuse_rope=True")
    return None


def _resolve_rope_tables(
    dummy_cache: Dict[tuple[torch.device, torch.dtype], torch.Tensor],
    reference: torch.Tensor,
    *,
    fuse_rope: bool,
    rotary_dim: Optional[int],
    table_dtype: Optional[torch.dtype] = None,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[int]]:
    """Normalize the optional public RoPE operands to the fixed kernel ABI."""
    table_dtype = reference.dtype if table_dtype is None else table_dtype
    if not fuse_rope:
        if rope_cos is not None or rope_sin is not None:
            raise ValueError("rope_cos and rope_sin require fuse_rope=True")
        key = (reference.device, table_dtype)
        dummy = dummy_cache.get(key)
        if dummy is None:
            dummy = torch.empty((1, 1), device=reference.device, dtype=table_dtype)
            dummy_cache[key] = dummy
        return dummy, dummy, None

    if rope_cos is None or rope_sin is None:
        raise ValueError("fuse_rope=True requires both rope_cos and rope_sin")
    assert rotary_dim is not None
    if rope_cos.device != reference.device or rope_sin.device != reference.device:
        raise ValueError("rope_cos and rope_sin must be on the same device as q")
    if rope_cos.dtype != table_dtype or rope_sin.dtype != table_dtype:
        raise ValueError(
            f"rope_cos and rope_sin must have the attention output dtype ({table_dtype})"
        )
    if not rope_cos.is_contiguous() or not rope_sin.is_contiguous():
        raise ValueError("rope_cos and rope_sin must be contiguous")
    if rope_cos.ndim != 2 or rope_sin.ndim != 2:
        raise ValueError("rope_cos and rope_sin must be rank-2 tensors")
    if rope_cos.shape != rope_sin.shape:
        raise ValueError("rope_cos and rope_sin must have the same shape")
    expected_cols = rotary_dim // 2
    if rope_cos.shape[0] <= 0 or rope_cos.shape[1] != expected_cols:
        raise ValueError(
            "rope_cos and rope_sin must have shape "
            f"[max_position, {expected_cols}], got {tuple(rope_cos.shape)}"
        )
    return rope_cos, rope_sin, int(rope_cos.shape[0])


def _rope_specialized_cache_key(
    base_key: object,
    call: AttentionCall,
) -> object:
    """Extend a kernel cache key only when RoPE changes generated code."""
    if not call.fuse_rope:
        return base_key
    if isinstance(base_key, tuple):
        return (*base_key, call.max_position, call.rotary_dim, call.rope_layout)
    return (base_key, call.max_position, call.rotary_dim, call.rope_layout)


def _resolve_group_scales(
    identity_cache: Dict[torch.device, torch.Tensor],
    reference: torch.Tensor,
    batch: int,
    heads_kv: int,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    *,
    required: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve the stable ``[B, H_kv]`` scale ABI without hot-path allocation."""
    scales = (q_scale, k_scale, v_scale)
    if any(scale is None for scale in scales):
        if any(scale is not None for scale in scales):
            raise ValueError("q_scale, k_scale, and v_scale must be supplied together")
        if required:
            raise ValueError("this numeric format requires q_scale, k_scale, and v_scale")
        identity = identity_cache.get(reference.device)
        if identity is None:
            identity = torch.ones((batch, heads_kv), device=reference.device, dtype=torch.float32)
            identity_cache[reference.device] = identity
        return identity, identity, identity

    expected_shape = (batch, heads_kv)
    resolved = []
    for name, scale in zip(("q_scale", "k_scale", "v_scale"), scales, strict=True):
        assert scale is not None
        if scale.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {scale.device}")
        if scale.dtype != torch.float32:
            raise ValueError(f"{name} must have dtype torch.float32")
        if tuple(scale.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        resolved.append(scale)
    return resolved[0], resolved[1], resolved[2]


def _build_packed_prefill_kernel(
    kernel_map: Dict[str, Kernel],
    key: str,
    call: AttentionCall,
) -> Kernel:
    """Construct the packed-prefill implementation *key* names.

    Every implementation of the slot takes the same constructor, so one step
    builds any of them and no caller carries a per-implementation argument list.
    """
    return kernel_map[key](
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        max_seqlen_q=call.max_seqlen_q,
        max_seqlen_kv=call.max_seqlen_kv,
        dim=call.dim,
        is_causal=call.is_causal,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        window_size_left=call.window_size_left,
        window_size_right=call.window_size_right,
        fuse_rope=call.fuse_rope,
        max_position=call.max_position,
        rotary_dim=call.rotary_dim,
        rope_layout=call.rope_layout,
        accum_dtype=call.accum_dtype,
        tune=call.tune,
    )


class GroupedQueryAttentionPrefillDenseFwdOp(Op):
    """Fixed-shape GQA prefill. Public layout: BSHD."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: Optional[torch.dtype] = None,
        seq_len_kv: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        fuse_rope: bool = False,
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
    ) -> None:
        # Nothing downstream validates these: this op builds its kernel itself,
        # so a zero heads_kv would surface as ZeroDivisionError inside a region.
        _validate_gqa_dims(heads, heads_kv, dim)
        seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv
        _validate_positive(batch=batch, seq_len=seq_len, seq_len_kv=seq_len_kv)
        if is_causal and seq_len > seq_len_kv:
            raise ValueError("causal dense prefill requires seq_len <= seq_len_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.fuse_rope = fuse_rope
        self.rotary_dim = _validate_rope_config(dim, fuse_rope, rotary_dim, rope_layout)
        self.rope_layout = rope_layout
        if dtype is not None:
            _validate_attention_dtype(dtype)
        self.output_dtype = dtype
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        # Packed ranges for a batch of equal-length requests, per device. Not a
        # kernel cache: the dense implementations take the same packed call as
        # every other, and a fixed-shape request supplies its ranges.
        self._cu_seqlens: Dict[tuple[torch.device, int], torch.Tensor] = {}
        self._identity_scales: Dict[torch.device, torch.Tensor] = {}
        self._rope_dummy_cache: Dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_fp8_tensor_core_fwd_kernel": GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
            "gqa_prefill_dense_sliding_fwd_kernel": GQASlidingWindowFwdWgmmaPipelinedKernel,
            "gqa_prefill_fwd_kernel": GQAPrefillFwdKernel,
            "gqa_prefill_causal_fwd_kernel": GQAPrefillFwdWsPersistentCausalKernel,
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
        }

    def attention_call(
        self, dtype: torch.dtype, max_position: Optional[int] = None
    ) -> AttentionCall:
        """State what one fixed-shape call is: a uniform dense packed request."""
        is_fp8 = fp8_dtype() is not None and dtype == fp8_dtype()
        output_dtype = self.output_dtype
        if output_dtype is None:
            if is_fp8:
                raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
            output_dtype = dtype
        return AttentionCall(
            dtype=output_dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            dim=self.dim,
            max_seqlen_q=self.seq_len,
            max_seqlen_kv=self.seq_len_kv,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=is_fp8,
            is_uniform=True,
            fuse_rope=self.fuse_rope,
            max_position=max_position,
            rotary_dim=self.rotary_dim,
            rope_layout=self.rope_layout,
            tune=self.tune,
        )

    def _get_kernel(self, dtype: torch.dtype, max_position: Optional[int] = None) -> Kernel:
        """The dense prefill implementation this wrapper's calls land on."""
        if dtype != fp8_dtype():
            _validate_attention_dtype(dtype)
        call = self.attention_call(dtype, max_position)
        key = self.select_kernel_key(DENSE_PREFILL_KEYS, call)

        def build() -> Kernel:
            return _build_packed_prefill_kernel(self.kernel_map, key, call)

        cache_key = _rope_specialized_cache_key(
            (dtype, call.dtype),
            call,
        )
        return self.get_or_build_kernel(key, key=cache_key, build=build)

    def _uniform_cu_seqlens(self, device: torch.device, seq_len: int) -> torch.Tensor:
        cache_key = (device, seq_len)
        cu_seqlens = self._cu_seqlens.get(cache_key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(self.batch + 1, device=device, dtype=torch.int32) * seq_len
            self._cu_seqlens[cache_key] = cu_seqlens
        return cu_seqlens

    def _scales_or_identity(
        self,
        q: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _resolve_group_scales(
            self._identity_scales,
            q,
            self.batch,
            self.heads_kv,
            q_scale,
            k_scale,
            v_scale,
            required=q.dtype == fp8_dtype(),
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        q_scale_shape: Optional[tuple[int, ...]] = None,
        k_scale_shape: Optional[tuple[int, ...]] = None,
        v_scale_shape: Optional[tuple[int, ...]] = None,
        rope_cos_shape: Optional[tuple[int, ...]] = None,
        rope_sin_shape: Optional[tuple[int, ...]] = None,
    ) -> Dict[str, tuple[int, ...]]:
        return {"o": tuple(q_shape)}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run fixed-shape GQA prefill."""
        expected_q = (self.batch, self.seq_len, self.heads, self.dim)
        expected_kv = (self.batch, self.seq_len_kv, self.heads_kv, self.dim)
        if tuple(q.shape) != expected_q:
            raise ValueError(f"q must have shape {expected_q}, got {tuple(q.shape)}")
        if tuple(k.shape) != expected_kv:
            raise ValueError(f"k must have shape {expected_kv}, got {tuple(k.shape)}")
        if tuple(v.shape) != expected_kv:
            raise ValueError(f"v must have shape {expected_kv}, got {tuple(v.shape)}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must have the same dtype")
        is_fp8 = fp8_dtype() is not None and q.dtype == fp8_dtype()
        if not is_fp8:
            _validate_attention_dtype(q.dtype)
            if self.output_dtype is not None and self.output_dtype != q.dtype:
                raise ValueError("16-bit prefill output dtype must match q/k/v dtype")
        q_scale, k_scale, v_scale = self._scales_or_identity(q, q_scale, k_scale, v_scale)
        rope_cos, rope_sin, max_position = _resolve_rope_tables(
            self._rope_dummy_cache,
            q,
            fuse_rope=self.fuse_rope,
            rotary_dim=self.rotary_dim,
            table_dtype=self.output_dtype or q.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        self.dtype = q.dtype

        # A fixed-shape request is a uniform packed one; packing is a view, so
        # the wrapper reaches the packed prefill call rather than handing BSHD
        # tensors to a kernel whose signature is packed.
        cu_seqlens_q = self._uniform_cu_seqlens(q.device, self.seq_len)
        cu_seqlens_kv = self._uniform_cu_seqlens(q.device, self.seq_len_kv)
        output = self._get_kernel(q.dtype, max_position)(
            q.view(-1, self.heads, self.dim),
            k.view(-1, self.heads_kv, self.dim),
            v.view(-1, self.heads_kv, self.dim),
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        return output.view(q.shape)


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
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: Optional[torch.dtype] = None,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        fuse_rope: bool = False,
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        _validate_positive(batch=batch, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_causal = is_causal
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.fuse_rope = fuse_rope
        self.rotary_dim = _validate_rope_config(dim, fuse_rope, rotary_dim, rope_layout)
        self.rope_layout = rope_layout
        if dtype is not None:
            _validate_attention_dtype(dtype)
        self.output_dtype = dtype
        self.validate_inputs = validate_inputs
        self._roofline_kwargs = None
        self._identity_scales: Dict[torch.device, torch.Tensor] = {}
        self._rope_dummy_cache: Dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def attention_call(
        self, input_dtype: torch.dtype, max_position: Optional[int] = None
    ) -> AttentionCall:
        """Describe a packed Varlen call without inspecting range values."""
        is_fp8 = fp8_dtype() is not None and input_dtype == fp8_dtype()
        output_dtype = self.output_dtype
        if output_dtype is None:
            if is_fp8:
                raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
            output_dtype = input_dtype
        return AttentionCall(
            dtype=output_dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            dim=self.dim,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=is_fp8,
            is_uniform=False,
            fuse_rope=self.fuse_rope,
            max_position=max_position,
            rotary_dim=self.rotary_dim,
            rope_layout=self.rope_layout,
            tune=self.tune,
        )

    def _get_kernel(self, dtype: torch.dtype, max_position: Optional[int] = None) -> Kernel:
        if dtype != fp8_dtype():
            _validate_attention_dtype(dtype)
        call = self.attention_call(dtype, max_position)
        key = self.select_kernel_key(VARLEN_PREFILL_KEYS, call)

        def build() -> Kernel:
            return _build_packed_prefill_kernel(self.kernel_map, key, call)

        cache_key = _rope_specialized_cache_key(
            (dtype, call.dtype),
            call,
        )
        return self.get_or_build_kernel(key, key=cache_key, build=build)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_varlen_fp8_tensor_core_fwd_kernel": GQAPrefillVarlenFP8TensorCoreFwdKernel,
            "gqa_sliding_window_varlen_fwd_kernel": GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
            "gqa_prefill_varlen_fwd_kernel": GQAPrefillVarlenFwdKernel,
        }

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

        # q carries the element type; k and v must agree with it.
        is_fp8 = fp8_dtype() is not None and tensors["q"].dtype == fp8_dtype()
        if not is_fp8:
            _validate_attention_dtype(tensors["q"].dtype)
            if self.output_dtype is not None and self.output_dtype != tensors["q"].dtype:
                raise ValueError("16-bit prefill output dtype must match q/k/v dtype")

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
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.dtype != tensors["q"].dtype:
                raise ValueError(f"Expected {name}.dtype {tensors['q'].dtype}, got {tensor.dtype}")

        for name in ("cu_seqlens_q", "cu_seqlens_kv"):
            tensor = tensors[name]
            expected_shape = (self.batch + 1,)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Expected {name} shape {expected_shape}, got {tuple(tensor.shape)}"
                )
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
            raise ValueError(f"cu_seqlens_q[-1] ({cu_q[-1]}) must equal q.shape[0] ({q.shape[0]})")
        if cu_kv[-1] != k.shape[0]:
            raise ValueError(
                f"cu_seqlens_kv[-1] ({cu_kv[-1]}) must equal k.shape[0] ({k.shape[0]})"
            )
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
            # Not _validate_positive: that names one scalar parameter after the
            # caller's own kwarg, while these are per-request lengths derived
            # from a tensor, reported for the set rather than for a parameter.
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
                f"sequence length ({actual_max_q})"
            )
        if self.max_seqlen_kv < actual_max_kv:
            raise ValueError(
                f"max_seqlen_kv ({self.max_seqlen_kv}) must be >= actual max KV "
                f"sequence length ({actual_max_kv})"
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._validate_forward_inputs(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        q_scale, k_scale, v_scale = _resolve_group_scales(
            self._identity_scales,
            q,
            self.batch,
            self.heads_kv,
            q_scale,
            k_scale,
            v_scale,
            required=q.dtype == fp8_dtype(),
        )
        rope_cos, rope_sin, max_position = _resolve_rope_tables(
            self._rope_dummy_cache,
            q,
            fuse_rope=self.fuse_rope,
            rotary_dim=self.rotary_dim,
            table_dtype=self.output_dtype or q.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        self.dtype = q.dtype
        output = self._get_kernel(q.dtype, max_position)(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        self._roofline_kwargs = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch": self.batch,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_kv": self.max_seqlen_kv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "is_causal": self.is_causal,
            "window_size_left": self.window_size_left,
            "window_size_right": self.window_size_right,
            "dtype": self.dtype,
            "output_dtype": self.output_dtype or self.dtype,
        }
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call"
            )
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
        max_seqlen_q: int,
        is_causal: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
        dtype: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        append_kv: bool = True,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        fuse_rope: bool = False,
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        rotary_dim = _validate_rope_config(dim, fuse_rope, rotary_dim, rope_layout)
        _validate_positive(
            batch=batch,
            max_pages_per_req=max_pages_per_req,
            page_size=page_size,
            max_seqlen_q=max_seqlen_q,
        )
        if page_size & (page_size - 1) != 0:
            raise ValueError("page_size must be a power of two")
        cache_dtype = _paged_cache_dtype(cache_dtype)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.groups = heads // heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.max_cache_len = max_pages_per_req * page_size
        self.dim = dim
        self.max_seqlen_q = max_seqlen_q
        self.is_causal = is_causal
        # None means the cache holds whatever element type forward is given.
        self.cache_dtype = cache_dtype
        if dtype is not None:
            _validate_attention_dtype(dtype)
        self.output_dtype = dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        if not isinstance(append_kv, bool):
            raise TypeError("append_kv must be a bool")
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.append_kv = append_kv
        self.validate_inputs = validate_inputs
        self.fuse_rope = fuse_rope
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self._rope_dummy_cache: Dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._identity_scales: Dict[torch.device, torch.Tensor] = {}

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_paged_native_fp8_tensor_core_fwd_kernel": GQAPrefillPagedNativeFP8TensorCoreFwdKernel,
            "gqa_prefill_paged_with_kv_cache_fwd_kernel": GQAPrefillPagedWithKVCacheFwdKernel,
            "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel": GQAPrefillPagedWithFP8KVCacheFwdKernel,
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel": GQAPrefillPagedWithKVCacheRopeFwdKernel,
        }

    def _resolved_cache_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Cache element type for an attention element type of *dtype*."""
        return dtype if self.cache_dtype is None else self.cache_dtype

    def attention_call(
        self, input_dtype: torch.dtype, max_position: Optional[int] = None
    ) -> AttentionCall:
        """State what one paged prefill call is, for selection to filter against."""
        is_fp8 = fp8_dtype() is not None and input_dtype == fp8_dtype()
        output_dtype = self.output_dtype
        if output_dtype is None:
            if is_fp8:
                raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
            output_dtype = input_dtype
        return AttentionCall(
            dtype=output_dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            dim=self.dim,
            max_seqlen_q=self.max_seqlen_q,
            max_pages_per_req=self.max_pages_per_req,
            page_size=self.page_size,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=is_fp8,
            cache_dtype=self._resolved_cache_dtype(input_dtype),
            append_kv=self.append_kv,
            fuse_rope=self.fuse_rope,
            max_position=max_position,
            rotary_dim=self.rotary_dim,
            rope_layout=self.rope_layout,
            tune=self.tune,
        )

    def _get_kernel(self, key: str, call: AttentionCall) -> Kernel:
        """The implementation *key* names, built once per element type.

        Every implementation of paged prefill takes the same constructor, so
        there is one build here and no per-implementation argument list.
        """

        def build() -> Kernel:
            return self.kernel_map[key](
                batch=call.batch,
                heads=call.heads,
                heads_kv=call.heads_kv,
                max_pages_per_req=call.max_pages_per_req,
                page_size=call.page_size,
                dim=call.dim,
                is_causal=call.is_causal,
                dtype=call.dtype,
                sm_scale=call.sm_scale,
                softcap=call.softcap,
                window_size_left=call.window_size_left,
                window_size_right=call.window_size_right,
                append_kv=call.append_kv,
                fuse_rope=call.fuse_rope,
                max_position=call.max_position,
                rotary_dim=call.rotary_dim,
                rope_layout=call.rope_layout,
                tune=call.tune,
            )

        cache_key = _rope_specialized_cache_key(call.dtype, call)
        return self.get_or_build_kernel(key, key=cache_key, build=build)

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        q_scale: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_position: Optional[int],
    ) -> None:
        tensors = {
            "q": q,
            "k_new": k_new,
            "v_new": v_new,
            "k_pages": k_pages,
            "v_pages": v_pages,
            "q_scale": q_scale,
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
                f"q must have shape [total_q, {self.heads}, {self.dim}], got {q.shape}"
            )
        if k_new.ndim != 3 or tuple(k_new.shape[1:]) != expected_kv_shape_tail:
            raise ValueError(
                f"k_new must have shape [total_q, {self.heads_kv}, {self.dim}], got {k_new.shape}"
            )
        if v_new.shape != k_new.shape:
            raise ValueError(
                f"v_new must have the same shape as k_new, got {v_new.shape} and {k_new.shape}"
            )
        if k_new.shape[0] != q.shape[0]:
            raise ValueError(
                f"k_new.shape[0] ({k_new.shape[0]}) must equal q.shape[0] ({q.shape[0]})"
            )
        if k_pages.ndim != 3 or tuple(k_pages.shape[1:]) != expected_kv_shape_tail:
            raise ValueError(
                f"k_pages must have shape [physical_tokens, {self.heads_kv}, {self.dim}], "
                f"got {k_pages.shape}"
            )
        if v_pages.shape != k_pages.shape:
            raise ValueError(
                f"v_pages must have the same shape as k_pages, got {v_pages.shape} and "
                f"{k_pages.shape}"
            )
        if k_pages.shape[0] % self.page_size != 0:
            raise ValueError("k_pages physical token dimension must be divisible by page_size")
        expected_scale_shape = (self.batch, self.heads_kv)
        for name, scale in (("q_scale", q_scale), ("k_scale", k_scale), ("v_scale", v_scale)):
            if tuple(scale.shape) != expected_scale_shape:
                raise ValueError(f"{name} must have shape {expected_scale_shape}")
        if cu_seqlens_q.shape != (self.batch + 1,):
            raise ValueError(
                f"cu_seqlens_q shape must be ({self.batch + 1},), got {tuple(cu_seqlens_q.shape)}"
            )
        if cache_seqlens.shape != (self.batch,):
            raise ValueError(
                f"cache_seqlens shape must be ({self.batch},), got {tuple(cache_seqlens.shape)}"
            )
        if block_table.shape != (self.batch, self.max_pages_per_req):
            raise ValueError(
                f"block_table shape must be ({self.batch}, {self.max_pages_per_req}), "
                f"got {tuple(block_table.shape)}"
            )

        # q carries the attention element type; k_new / v_new must agree with it.
        is_fp8 = fp8_dtype() is not None and q.dtype == fp8_dtype()
        if not is_fp8:
            _validate_attention_dtype(q.dtype)
            if self.output_dtype is not None and self.output_dtype != q.dtype:
                raise ValueError("16-bit prefill output dtype must match q/k_new/v_new dtype")
        cache_dtype = self._resolved_cache_dtype(q.dtype)
        fp8_cache_dtype = fp8_dtype()
        if cache_dtype != q.dtype and cache_dtype != fp8_cache_dtype:
            raise ValueError(
                "cache_dtype must be either same as the q element type or "
                f"torch.float8_e4m3fn, got {cache_dtype}"
            )
        for name, tensor in [("k_new", k_new), ("v_new", v_new)]:
            if tensor.dtype != q.dtype:
                raise ValueError(f"Expected {name}.dtype {q.dtype}, got {tensor.dtype}")
        for name, tensor in [("k_pages", k_pages), ("v_pages", v_pages)]:
            if tensor.dtype != cache_dtype:
                raise ValueError(f"Expected {name}.dtype {cache_dtype}, got {tensor.dtype}")
        for name, tensor in [("q_scale", q_scale), ("k_scale", k_scale), ("v_scale", v_scale)]:
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
        for name, tensor in [
            ("cu_seqlens_q", cu_seqlens_q),
            ("cache_seqlens", cache_seqlens),
            ("block_table", block_table),
        ]:
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")

        # Value-level validation synchronizes device data back to Python. Keep
        # it available as an explicit diagnostic mode, never on the serving
        # hot path used by the default Op contract.
        if not self.validate_inputs:
            return

        if is_fp8 or cache_dtype == fp8_cache_dtype:
            scaled_tensors = (
                (("q_scale", q_scale), ("k_scale", k_scale), ("v_scale", v_scale))
                if is_fp8
                else (("k_scale", k_scale), ("v_scale", v_scale))
            )
            for name, tensor in scaled_tensors:
                if not torch.all(torch.isfinite(tensor) & (tensor > 0)).item():
                    raise ValueError(f"{name} must contain finite positive values")

        if int(cu_seqlens_q[0].item()) != 0:
            raise ValueError("cu_seqlens_q[0] must be 0")
        q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        if torch.any(q_lens < 0).item():
            raise ValueError("cu_seqlens_q must be non-decreasing")
        total_q = int(cu_seqlens_q[-1].item())
        if total_q != q.shape[0]:
            raise ValueError(f"cu_seqlens_q[-1] ({total_q}) must equal q.shape[0] ({q.shape[0]})")
        actual_max_q = int(q_lens.max().item())
        if self.max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({self.max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})"
            )

        min_cache_len = int(cache_seqlens.min().item())
        max_total_len = int((cache_seqlens + q_lens).max().item())
        if min_cache_len < 0:
            raise ValueError("cache_seqlens must be non-negative")
        if max_total_len > self.max_cache_len:
            raise ValueError(
                "cache_seqlens + q_len exceeds paged KV capacity: "
                f"max total length {max_total_len}, capacity {self.max_cache_len}"
            )
        if self.fuse_rope and max_position is not None and max_total_len > max_position:
            raise ValueError(
                "cache_seqlens + q_len exceeds RoPE max_position: "
                f"max total length {max_total_len}, max_position {max_position}"
            )

        num_pages = k_pages.shape[0] // self.page_size
        min_page = int(block_table.min().item())
        max_page = int(block_table.max().item())
        if min_page < 0:
            raise ValueError("block_table must contain non-negative physical page ids")
        if max_page >= num_pages:
            raise ValueError(
                f"block_table references page {max_page}, but only {num_pages} pages exist"
            )

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cache_dtype = self._resolved_cache_dtype(q.dtype)
        q_scale, k_scale, v_scale = _resolve_group_scales(
            self._identity_scales,
            q,
            self.batch,
            self.heads_kv,
            q_scale,
            k_scale,
            v_scale,
            required=q.dtype == fp8_dtype() or cache_dtype == fp8_dtype(),
        )
        rope_cos, rope_sin, max_position = _resolve_rope_tables(
            self._rope_dummy_cache,
            q,
            fuse_rope=self.fuse_rope,
            rotary_dim=self.rotary_dim,
            table_dtype=self.output_dtype or q.dtype,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        self._validate_forward_inputs(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            q_scale,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_position,
        )
        self.dtype = q.dtype
        call = self.attention_call(q.dtype, max_position)
        key = self.select_kernel_key(PAGED_PREFILL_KEYS, call)
        return self._get_kernel(key, call)(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            q_scale,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            self.max_seqlen_q,
            rope_cos,
            rope_sin,
        )

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time."
        )

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for paged varlen ops; "
            "compute per-sample from cu_seqlens and cache_seqlens at call time."
        )


class GroupedQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernels(self, dtype: torch.dtype) -> tuple[Kernel, Kernel]:
        """Return (preprocess, backward) kernels for *dtype*, building once each."""

        def build_preprocess() -> Kernel:
            return self.kernel_map["gqa_bwd_preprocess_kernel"](
                self.batch,
                self.heads,
                self.seq_len,
                self.dim,
                dtype,
                tune=self.tune,
            )

        def build_backward() -> Kernel:
            return self.kernel_map["gqa_bwd_kernel"](
                self.batch,
                self.heads,
                self.heads_kv,
                self.seq_len,
                self.dim,
                self.is_causal,
                dtype,
                tune=self.tune,
            )

        return (
            self.get_or_build_kernel(
                "gqa_bwd_preprocess_kernel", key=dtype, build=build_preprocess
            ),
            self.get_or_build_kernel("gqa_bwd_kernel", key=dtype, build=build_backward),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel": GQABwdWgmmaPipelinedKernel,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()
        self._validate_dtypes(q, k, v, o, do, lse)
        self.dtype = q.dtype
        prep_kernel, kernel = self._get_kernels(q.dtype)
        delta = prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(q.dtype)
        dk, dv = dk.to(q.dtype), dv.to(q.dtype)
        return dq, dk, dv


class GroupedQueryAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        _validate_attention_dtype(dtype)
        call = self.attention_call(dtype)
        key = self.select_kernel_key(DECODE_KEYS, call)

        def build() -> Kernel:
            return self.kernel_map[key](
                call.batch,
                call.heads,
                call.heads_kv,
                call.seqlen_kv,
                call.dim,
                call.dtype,
                sm_scale=call.sm_scale,
                softcap=call.softcap,
                tune=call.tune,
            )

        return self.get_or_build_kernel(key, key=dtype, build=build)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_decode_kernel": GQADecodeKernel,
            "gqa_decode_bs1_kernel": GQADecodeBs1Kernel,
        }

    def attention_call(self, dtype: torch.dtype) -> AttentionCall:
        """State what one decode call is, for selection to filter candidates against.

        The element type arrives with the inputs, so it is a property of the call
        rather than of the op: one instance serves every dtype it is handed.
        """
        return AttentionCall(
            dtype=dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            seqlen_kv=self.seqlen_kv,
            dim=self.dim,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            tune=self.tune,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode="constant", value=0
            )
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode="constant", value=0
            )

        self._validate_dtypes(q, k, v)
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k, v, real_seqlen_kv)


class GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged GQA decode with dynamic KV cache. Layout: Q [batch, heads, dim] (BHD);
    K, V physical cache [seqlen_kv, heads_kv, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        _validate_positive(page_size=page_size)
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        _validate_attention_dtype(dtype)
        call = self.attention_call(dtype)
        key = self.select_kernel_key(PAGED_DECODE_KEYS, call)

        def build() -> Kernel:
            return self.kernel_map[key](
                call.batch,
                call.heads,
                call.heads_kv,
                call.seqlen_kv,
                call.dim,
                call.page_size,
                call.dtype,
                sm_scale=call.sm_scale,
                softcap=call.softcap,
                tune=call.tune,
            )

        return self.get_or_build_kernel(key, key=dtype, build=build)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_decode_paged_kernel": GQADecodePagedKernel,
            "gqa_decode_paged_bs1_kernel": GQADecodePagedBs1Kernel,
        }

    def attention_call(self, dtype: torch.dtype) -> AttentionCall:
        """State what one paged decode call is, for selection to filter against."""
        return AttentionCall(
            dtype=dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            seqlen_kv=self.seqlen_kv,
            dim=self.dim,
            page_size=self.page_size,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            tune=self.tune,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        self.dtype = q.dtype
        return self._get_kernel(q.dtype)(q, k, v, real_seqlen_kv, block_table)
