import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Callable, ClassVar, Dict, Optional

import torch
import torch.nn.functional as F

from tileops.backend import Target
from tileops.kernels.attention import (
    FlashAttnBwdPreprocessKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodePagedBs1Kernel,
    GQADecodePagedKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
    GQAFwdWsPersistentCausalKernel,
    GQAPrefillDenseFwdKernel,
    GQAPrefillFwdWsPersistentCausalKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.attention.prefill import DensePrefillKernel
from tileops.kernels.kernel_base import Kernel

from ..compile_boundary import get_instance
from ..op_base import Op
from ..rope import base_freqs
from .selection import (
    DECODE_KEYS,
    DENSE_PREFILL_KEYS,
    PACKED_PREFILL_KEYS,
    PAGED_DECODE_KEYS,
    PAGED_PREFILL_KEYS,
    AttentionCall,
    check_packed_prefill_request,
    fp8_dtype,
)

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillDenseFwdOp",
    "GroupedQueryAttentionPrefillFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
]

_ROPE_LAYOUTS = frozenset(("neox", "interleaved"))
_POS_ENCODING_MODES = frozenset(("none", "rope"))
_DENSE_SPECIALIZATION_CACHE_SIZE = 16
_DENSE_ROPE_CACHE_SIZE = 8


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


def _validate_same_device(reference: torch.Tensor, **tensors: torch.Tensor) -> None:
    """Enforce the target-neutral one-device contract for manifest inputs."""
    for name, tensor in tensors.items():
        if tensor.device != reference.device:
            raise ValueError(
                f"{name} must be on the same device as q ({reference.device}), got {tensor.device}"
            )


def _attention_scale(dim: int, sm_scale: Optional[float]) -> float:
    scale = dim**-0.5 if sm_scale is None else sm_scale
    if not math.isfinite(scale):
        raise ValueError("sm_scale must be finite")
    return scale


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
    pos_encoding_mode: str,
    rotary_dim: Optional[int],
    rope_layout: str,
    rope_base: float,
) -> Optional[int]:
    """Validate static RoPE semantics shared by every prefill topology."""
    if pos_encoding_mode not in _POS_ENCODING_MODES:
        raise ValueError(
            "pos_encoding_mode must be one of "
            f"{sorted(_POS_ENCODING_MODES)}, got {pos_encoding_mode!r}"
        )
    if rope_layout not in _ROPE_LAYOUTS:
        raise ValueError(f"rope_layout must be one of {sorted(_ROPE_LAYOUTS)}, got {rope_layout!r}")
    if not math.isfinite(rope_base) or rope_base <= 0:
        raise ValueError("rope_base must be finite and positive")
    if pos_encoding_mode == "rope":
        return _rope_rotary_dim(dim, rotary_dim)
    if rotary_dim is not None:
        raise ValueError("rotary_dim requires pos_encoding_mode='rope'")
    return None


def _prepare_group_scales(
    reference: torch.Tensor,
    batch: int,
    heads_kv: int,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Validate supplied scales while preserving optional-input presence."""
    scales = (q_scale, k_scale, v_scale)
    if any(scale is None for scale in scales):
        if any(scale is not None for scale in scales):
            raise ValueError("q_scale, k_scale, and v_scale must be supplied together")
        return ()

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
        resolved.append(scale.contiguous())
    return tuple(resolved)


def _build_prefill_kernel(
    kernel_map: Dict[str, Kernel],
    key: str,
    call: AttentionCall,
) -> Kernel:
    """Construct the prefill implementation *key* names.

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


@dataclass
class _DenseRopeTables:
    cos: torch.Tensor
    sin: torch.Tensor
    ready: Optional[torch.cuda.Event]


@dataclass
class _DensePrefillBuiltin:
    """Backend callable that dispatches each Dense shape to one native kernel."""

    kernel_map: Dict[str, Kernel]
    select_kernel_key: Callable[[tuple[str, ...], object], str]
    device: torch.device
    arch: int
    h200: bool
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    is_causal: bool
    sm_scale: Optional[float]
    softcap: float
    window_size_left: int
    window_size_right: int
    fuse_rope: bool
    rotary_dim: Optional[int]
    rope_layout: str
    rope_base: float
    tune_enabled: Callable[[], bool]
    _kernel_cache: OrderedDict[AttentionCall, DensePrefillKernel] = field(
        default_factory=OrderedDict, repr=False
    )
    # Kept as a list so Op._entry_kernels() can enumerate and autotune the
    # concrete kernels owned by this callable. The OrderedDict above is the
    # lookup/index; both containers are updated together under _cache_lock.
    _retained_kernels: list[DensePrefillKernel] = field(default_factory=list, repr=False)
    _rope_table_cache: OrderedDict[tuple[int, int], _DenseRopeTables] = field(
        default_factory=OrderedDict, repr=False
    )
    # CUDA Graphs capture raw device pointers, not Python Tensor ownership.
    # Keep tables used by a capture alive even after the ordinary bounded memo
    # evicts their lookup entry. Captured graph signatures are intentionally
    # pinned for this callable's lifetime.
    _captured_rope_tables: dict[tuple[int, int], _DenseRopeTables] = field(
        default_factory=dict, repr=False
    )
    _cache_lock: RLock = field(default_factory=RLock, repr=False)

    def _selection_facts(self, q: torch.Tensor, k: torch.Tensor) -> AttentionCall:
        batch, seq_len_q, heads, dim = q.shape
        _, seq_len_kv, heads_kv, _ = k.shape
        resolved_rotary_dim = _validate_rope_config(
            dim,
            "rope" if self.fuse_rope else "none",
            self.rotary_dim,
            self.rope_layout,
            self.rope_base,
        )
        return AttentionCall(
            arch=self.arch,
            h200=self.h200,
            dtype=self.output_dtype,
            prefill_topology="dense",
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_seqlen_q=seq_len_q,
            max_seqlen_kv=seq_len_kv,
            is_causal=self.is_causal,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=self.input_dtype == fp8_dtype(),
            is_uniform=True,
            fuse_rope=self.fuse_rope,
            max_position=seq_len_kv if self.fuse_rope else None,
            rotary_dim=resolved_rotary_dim,
            rope_layout=self.rope_layout,
            tune=self.tune_enabled(),
        )

    def _rope_tables(self, call: AttentionCall) -> tuple[torch.Tensor, torch.Tensor]:
        max_position = call.max_position or 1
        rotary_dim = call.rotary_dim or 0
        key = (max_position, rotary_dim)
        is_capturing = self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        cached = self._rope_table_cache.get(key)
        if cached is not None:
            if self.device.type == "cuda" and not is_capturing:
                stream = torch.cuda.current_stream(self.device)
                if cached.ready is not None and not cached.ready.query():
                    stream.wait_event(cached.ready)
                cached.cos.record_stream(stream)
                cached.sin.record_stream(stream)
            if is_capturing:
                self._captured_rope_tables.setdefault(key, cached)
            return cached.cos, cached.sin
        if is_capturing:
            raise RuntimeError(
                "Dense prefill CUDA Graph capture requires a same-signature warmup"
            )
        with self._cache_lock:
            # Recheck after acquiring: another host thread may have filled the
            # miss while this one waited. Hot CUDA-graph hits never take a lock.
            cached = self._rope_table_cache.get(key)
            if cached is not None:
                if self.device.type == "cuda":
                    stream = torch.cuda.current_stream(self.device)
                    if cached.ready is not None and not cached.ready.query():
                        stream.wait_event(cached.ready)
                    cached.cos.record_stream(stream)
                    cached.sin.record_stream(stream)
                return cached.cos, cached.sin
            if call.fuse_rope:
                assert call.rotary_dim is not None
                cos, sin = _base_freqs(
                    call.rotary_dim,
                    max_position,
                    base=self.rope_base,
                    dtype=self.output_dtype,
                    device=self.device,
                )
            else:
                dummy = torch.empty((1, 1), device=self.device, dtype=self.output_dtype)
                cos, sin = dummy, dummy
            ready = None
            if self.device.type == "cuda":
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.device))
            cached = _DenseRopeTables(cos=cos, sin=sin, ready=ready)
            self._rope_table_cache[key] = cached
            if len(self._rope_table_cache) > _DENSE_ROPE_CACHE_SIZE:
                self._rope_table_cache.popitem(last=False)
            return cached.cos, cached.sin

    def _kernel_for(self, call: AttentionCall) -> DensePrefillKernel:
        # Tuning is lifecycle state, not part of the specialization identity.
        # op.autotune() tunes retained kernels and flips tune_enabled() for
        # specializations constructed after that point.
        signature = replace(call, tune=False)
        cached = self._kernel_cache.get(signature)
        if cached is not None:
            return cached
        with self._cache_lock:
            # Miss-only serialization keeps construction unique without putting
            # a lock on the CUDA-graph capture/replay hit path.
            cached = self._kernel_cache.get(signature)
            if cached is not None:
                return cached

            key = self.select_kernel_key(DENSE_PREFILL_KEYS, call)
            kernel = _build_prefill_kernel(self.kernel_map, key, call)
            if not isinstance(kernel, DensePrefillKernel):
                raise TypeError(
                    f"Dense prefill selected a non-Dense kernel: {type(kernel).__name__}"
                )
            self._kernel_cache[signature] = kernel
            self._retained_kernels.append(kernel)
            if len(self._kernel_cache) > _DENSE_SPECIALIZATION_CACHE_SIZE:
                _, evicted = self._kernel_cache.popitem(last=False)
                self._retained_kernels = [
                    retained for retained in self._retained_kernels if retained is not evicted
                ]
            return kernel

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *scales: torch.Tensor,
    ) -> torch.Tensor:
        if len(scales) not in (0, 3):
            raise ValueError("Dense prefill expects either zero or three scale tensors")
        call = self._selection_facts(q, k)
        kernel = self._kernel_for(call)
        rope_cos, rope_sin = self._rope_tables(call)
        return kernel(
            q,
            k,
            v,
            *scales,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )


class GroupedQueryAttentionPrefillDenseFwdOp(Op):
    """Shape-agnostic BSHD GQA prefill with constructor-owned position encoding.

    ``dtype=None`` has the complete public meaning "follow ``q.dtype``" for
    float16/bfloat16 calls; it is not a deferred backend choice.  A backend
    derives that result type from the first :class:`TensorSpec` it receives.
    FP8 input has no such unique result type, so those calls must select an
    explicit float16 or bfloat16 ``dtype`` at construction.
    """

    compile_op_names: ClassVar[tuple[str, ...]] = ("top::gqa_prefill_dense_fwd",)

    def __init__(
        self,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
        pos_encoding_mode: str = "none",
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        rope_base: float = 10000.0,
        target: Target = None,
    ) -> None:
        if pos_encoding_mode not in ("none", "rope"):
            raise ValueError(f"pos_encoding_mode must be 'none' or 'rope', got {pos_encoding_mode}")

        if rotary_dim is not None and pos_encoding_mode != "rope":
            raise ValueError("rotary_dim requires pos_encoding_mode='rope'")

        if pos_encoding_mode == "rope" and (rope_base <= 0 or not math.isfinite(rope_base)):
            raise ValueError("rope_base must be finite and positive")

        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")

        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)

        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right

        self.pos_encoding_mode = pos_encoding_mode
        self.fuse_rope = pos_encoding_mode == "rope"
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.rope_base = rope_base

        if dtype is not None:
            _validate_attention_dtype(dtype)
        self.dtype = dtype
        self.output_dtype = dtype

        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

        # Reporting state is deliberately separate from kernel dispatch. It is
        # populated only after a successful call because eval_roofline() has no
        # input arguments in the current Op protocol.
        self._roofline_state: Optional[dict[str, object]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_fp8_tensor_core_fwd_kernel": GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
            "gqa_prefill_dense_sliding_fwd_kernel": GQASlidingWindowFwdWgmmaPipelinedKernel,
            "gqa_prefill_dense_fwd_kernel": GQAPrefillDenseFwdKernel,
            "gqa_prefill_causal_fwd_kernel": GQAPrefillFwdWsPersistentCausalKernel,
            "gqa_prefill_square_fwd_kernel": GQAFwdWsPersistentCausalKernel,
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Validate the manifest dtype union before target dispatch."""
        fp8 = fp8_dtype()
        allowed_inputs = {torch.float16, torch.bfloat16}
        if fp8 is not None:
            allowed_inputs.add(fp8)
        if q.dtype not in allowed_inputs:
            raise ValueError(f"q.dtype must be float16, bfloat16, or float8_e4m3fn, got {q.dtype}")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q/k/v must have the same dtype")

        is_fp8 = fp8 is not None and q.dtype == fp8
        if is_fp8:
            # A real instance owns ``output_dtype``. The CPU manifest parity
            # probe deliberately bypasses ``__init__`` and therefore validates
            # only the input side of each listed dtype combo here.
            output_dtype = getattr(self, "output_dtype", None)
            if hasattr(self, "output_dtype") and output_dtype not in (
                torch.float16,
                torch.bfloat16,
            ):
                raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
        else:
            output_dtype = getattr(self, "output_dtype", None)
            if output_dtype is not None and output_dtype != q.dtype:
                raise ValueError("16-bit prefill output dtype must match q/k/v dtype")
            output_dtype = output_dtype or q.dtype

        scales = (q_scale, k_scale, v_scale)
        for name, scale in zip(("q_scale", "k_scale", "v_scale"), scales, strict=True):
            if scale is not None and scale.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")

    def _build_builtin(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> _DensePrefillBuiltin:
        """Build one shape-agnostic callable for this builtin target and dtype."""
        if dtype != fp8_dtype():
            _validate_attention_dtype(dtype)
        output_dtype = self.output_dtype or dtype
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
        target_facts = AttentionCall.from_device(device)
        return _DensePrefillBuiltin(
            kernel_map=self.kernel_map,
            select_kernel_key=self.select_kernel_key,
            device=device,
            arch=target_facts.arch,
            h200=target_facts.h200,
            input_dtype=dtype,
            output_dtype=output_dtype,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            fuse_rope=self.fuse_rope,
            rotary_dim=self.rotary_dim,
            rope_layout=self.rope_layout,
            rope_base=self.rope_base,
            tune_enabled=lambda: self.tune,
        )

    def _get_entry(
        self,
        inputs: tuple[torch.Tensor, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> _DensePrefillBuiltin:
        dim = inputs[0].shape[-1]
        manifest_params = self._manifest_params()
        manifest_params.update(
            sm_scale=_attention_scale(dim, self.sm_scale),
            dtype=self.output_dtype or dtype,
            rotary_dim=(
                _validate_rope_config(
                    dim,
                    self.pos_encoding_mode,
                    self.rotary_dim,
                    self.rope_layout,
                    self.rope_base,
                )
                if self.fuse_rope
                else None
            ),
        )
        return self.get_or_build_kernel(
            "gqa_prefill_dense",
            inputs,
            key=(device, dtype, self.output_dtype),
            build=lambda: self._build_builtin(dtype, device),
            params=manifest_params,
        )

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        q_scale_shape: Optional[tuple[int, ...]] = None,
        k_scale_shape: Optional[tuple[int, ...]] = None,
        v_scale_shape: Optional[tuple[int, ...]] = None,
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
    ) -> torch.Tensor:
        """Run shape-agnostic GQA prefill behind the target-independent graph node."""
        return _gqa_prefill_dense_fwd(
            q,
            k,
            v,
            q_scale,
            k_scale,
            v_scale,
            self._instance_key,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run GQA prefill with shape inference."""
        B, S_q, H, D = q.shape
        B_k, S_kv, H_kv, D_k = k.shape

        if k.shape != v.shape:
            raise ValueError(f"k and v must have the same shape, got k={k.shape}, v={v.shape}")
        if B_k != B or D_k != D:
            raise ValueError(
                f"q and k must have matching batch and dim, got q=({B}, {S_q}, {H}, {D}), k=({B_k}, {S_kv}, {H_kv}, {D_k})"
            )

        _validate_positive(batch=B, seq_len_q=S_q, seq_len_kv=S_kv)

        _validate_gqa_dims(H, H_kv, D)

        if self.is_causal and S_q > S_kv:
            raise ValueError(
                f"causal dense prefill requires seq_len <= seq_len_kv, got {S_q} > {S_kv}"
            )

        if self.fuse_rope and S_q > S_kv:
            raise ValueError(
                f"fused RoPE uses bottom-right Q positions and requires seq_len <= seq_len_kv, got {S_q} > {S_kv}"
            )

        _attention_scale(D, self.sm_scale)
        _validate_rope_config(
            D, self.pos_encoding_mode, self.rotary_dim, self.rope_layout, self.rope_base
        )

        self._validate_dtypes(q, k, v, q_scale, k_scale, v_scale)

        has_scales = tuple(scale is not None for scale in (q_scale, k_scale, v_scale))
        if any(has_scales) and not all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale must be supplied together")
        is_fp8 = q.dtype == fp8_dtype()
        if is_fp8 and not all(has_scales):
            raise ValueError("FP8 input requires q_scale, k_scale, and v_scale")
        if not is_fp8 and all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale are only valid for FP8 input")

        _validate_same_device(q, k=k, v=v)
        scales = _prepare_group_scales(q, B, H_kv, q_scale, k_scale, v_scale)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        inputs = (q, k, v, *scales)
        entry = self._get_entry(inputs, q.dtype, q.device)
        output = entry(*inputs)
        self._roofline_state = {
            "q_shape": tuple(q.shape),
            "kv_shape": tuple(k.shape),
            "input_dtype": q.dtype,
            "output_dtype": self.output_dtype or q.dtype,
            "is_causal": self.is_causal,
            "window_size_left": self.window_size_left,
            "window_size_right": self.window_size_right,
            "fuse_rope": self.fuse_rope,
            "rotary_dim": self.rotary_dim or D,
            "max_position": S_kv if self.fuse_rope else None,
        }
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_state is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call"
            )
        from tileops.perf.formulas import gqa_fwd_roofline

        return gqa_fwd_roofline(**self._roofline_state)


@torch.library.custom_op("top::gqa_prefill_dense_fwd", mutates_args=())
def _gqa_prefill_dense_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    """Opaque Op-owned compile boundary; implementation dispatch stays untraced."""
    return get_instance(instance_key)._eager_forward(q, k, v, q_scale, k_scale, v_scale)


@_gqa_prefill_dense_fwd.register_fake
def _gqa_prefill_dense_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    instance_key: str,
) -> torch.Tensor:
    """Manifest-derived output metadata for the target-independent graph node."""
    op = get_instance(instance_key)
    output_dtype = op.output_dtype
    if output_dtype is None:
        if fp8_dtype() is not None and q.dtype == fp8_dtype():
            raise ValueError("dtype must select a float16 or bfloat16 output for FP8 input")
        output_dtype = q.dtype
    shapes = op._infer_output_shapes(
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        None if q_scale is None else tuple(q_scale.shape),
        None if k_scale is None else tuple(k_scale.shape),
        None if v_scale is None else tuple(v_scale.shape),
    )
    return q.new_empty(shapes["o"], dtype=output_dtype)


class GroupedQueryAttentionPrefillFwdOp(Op):
    """Deprecated packed GQA prefill compatibility surface. Layout: THD.

    Both uniform and ragged inputs execute a packed Varlen kernel. Fixed-shape
    BSHD, fixed sliding-window, and native-FP8 calls belong to
    :class:`GroupedQueryAttentionPrefillDenseFwdOp`. This compatibility Op is
    removed when its remaining public ABI moves to the canonical Varlen Op in
    #1917.
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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        _validate_gqa_dims(heads, heads_kv, dim)
        _validate_positive(batch=batch, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
        if is_causal and max_seqlen_q > max_seqlen_kv:
            raise ValueError("causal prefill requires max_seqlen_q <= max_seqlen_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}"
            )
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}"
            )
        if backend not in ("auto", "dense", "varlen", "fp8", "sliding_window"):
            raise ValueError(
                "backend must be one of 'auto', 'dense', 'varlen', 'fp8', or 'sliding_window'"
            )
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
        self.tune = tune
        self._roofline_kwargs = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_varlen_fwd_kernel": GQAPrefillVarlenFwdKernel,
            "gqa_sliding_window_varlen_fwd_kernel": GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
        }

    def attention_call(self, *, is_fp8: bool) -> AttentionCall:
        """State what one prefill call is, for selection to filter candidates against.

        Args:
            is_fp8: Whether the inputs carry ``torch.float8_e4m3fn`` elements.
        """
        return AttentionCall(
            dtype=self.dtype,
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
            backend=self.backend,
            is_fp8=is_fp8,
            is_uniform=False,
            tune=self.tune,
        )

    def _kernel_for(self, key: str, call: AttentionCall) -> Kernel:
        """The implementation *key* names, built once for this element type."""

        return self.get_or_build_kernel(
            key,
            key=call.dtype,
            build=lambda: _build_prefill_kernel(self.kernel_map, key, call),
        )

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
            raise ValueError(
                "Packed FP8 prefill moved to GroupedQueryAttentionPrefillDenseFwdOp; "
                "Varlen FP8 support is tracked by #1917."
            )
        else:
            if q.dtype != self.dtype or k.dtype != self.dtype or v.dtype != self.dtype:
                raise ValueError(f"q/k/v dtype must match op dtype {self.dtype}.")
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_kv.dtype != torch.int32:
            raise ValueError("cu_seqlens_q/cu_seqlens_kv must be torch.int32.")
        if (
            q_scale.dtype != torch.float32
            or k_scale.dtype != torch.float32
            or v_scale.dtype != torch.float32
        ):
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
                f"q must have shape [T, {self.heads}, {self.dim}], got {tuple(q.shape)}"
            )
        if k.ndim != 3 or tuple(k.shape[1:]) != (self.heads_kv, self.dim):
            raise ValueError(
                f"k must have shape [T, {self.heads_kv}, {self.dim}], got {tuple(k.shape)}"
            )
        if v.ndim != 3 or tuple(v.shape[1:]) != (self.heads_kv, self.dim):
            raise ValueError(
                f"v must have shape [T, {self.heads_kv}, {self.dim}], got {tuple(v.shape)}"
            )
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
            "dtype": q.dtype,
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
        self._validate_common_shapes(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale
        )
        call = self.attention_call(is_fp8=False)
        check_packed_prefill_request(call)
        key = self.select_kernel_key(PACKED_PREFILL_KEYS, call)
        output = self._kernel_for(key, call)(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale
        )
        self._record_roofline(q, k, cu_seqlens_q, cu_seqlens_kv)
        return output

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call"
            )
        from tileops.perf.formulas import gqa_prefill_varlen_fwd_roofline

        kwargs = dict(self._roofline_kwargs)
        kwargs["q_lens"] = GroupedQueryAttentionPrefillVarlenFwdOp._lengths_from_cu_seqlens(
            kwargs.pop("cu_seqlens_q")
        )
        kwargs["kv_lens"] = GroupedQueryAttentionPrefillVarlenFwdOp._lengths_from_cu_seqlens(
            kwargs.pop("cu_seqlens_kv")
        )
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
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
        self.validate_inputs = validate_inputs
        self._roofline_kwargs = None

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        _validate_attention_dtype(dtype)

        def build() -> Kernel:
            return self.kernel_map["gqa_prefill_varlen_fwd_kernel"](
                batch=self.batch,
                heads=self.heads,
                heads_kv=self.heads_kv,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_kv=self.max_seqlen_kv,
                dim=self.dim,
                is_causal=self.is_causal,
                dtype=dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                tune=self.tune,
            )

        return self.get_or_build_kernel("gqa_prefill_varlen_fwd_kernel", key=dtype, build=build)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_prefill_varlen_fwd_kernel": GQAPrefillVarlenFwdKernel}

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
        _validate_attention_dtype(tensors["q"].dtype)

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
    ) -> torch.Tensor:
        self._validate_forward_inputs(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        self.dtype = q.dtype
        output = self._get_kernel(q.dtype)(q, k, v, cu_seqlens_q, cu_seqlens_kv)
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
        is_causal: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
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
            _validate_positive(max_position=max_position)
        elif rotary_dim is not None:
            raise ValueError("rotary_dim requires fuse_rope=True")
        _validate_positive(batch=batch, max_pages_per_req=max_pages_per_req, page_size=page_size)
        if page_size & (page_size - 1) != 0:
            raise ValueError("page_size must be a power of two")
        cache_dtype = _paged_cache_dtype(cache_dtype)
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
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
        # None means the cache holds whatever element type forward is given.
        self.cache_dtype = cache_dtype
        self.sm_scale = _attention_scale(dim, sm_scale)
        self.softcap = _score_softcap(softcap)
        self.fuse_rope = fuse_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._rope_cos_cache: Dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_prefill_paged_with_kv_cache_fwd_kernel": GQAPrefillPagedWithKVCacheFwdKernel,
            "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel": GQAPrefillPagedWithFP8KVCacheFwdKernel,
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel": GQAPrefillPagedWithKVCacheRopeFwdKernel,
        }

    def _resolved_cache_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Cache element type for an attention element type of *dtype*."""
        return dtype if self.cache_dtype is None else self.cache_dtype

    def attention_call(self, dtype: torch.dtype) -> AttentionCall:
        """State what one paged prefill call is, for selection to filter against."""
        return AttentionCall(
            dtype=dtype,
            batch=self.batch,
            heads=self.heads,
            heads_kv=self.heads_kv,
            dim=self.dim,
            max_pages_per_req=self.max_pages_per_req,
            page_size=self.page_size,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            cache_dtype=self._resolved_cache_dtype(dtype),
            fuse_rope=self.fuse_rope,
            max_position=self.max_position,
            rotary_dim=self.rotary_dim,
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
                max_position=call.max_position,
                rotary_dim=call.rotary_dim,
                tune=call.tune,
            )

        return self.get_or_build_kernel(key, key=call.dtype, build=build)

    def _rope_tables(self, device: torch.device, dtype: torch.dtype):
        """Rotary tables for this op, or ``(None, None)`` when it fuses no RoPE."""
        if not self.fuse_rope:
            return None, None
        return self._get_rope_cos_sin(device, dtype)

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
        if k_scale.shape != (1,) or v_scale.shape != (1,):
            raise ValueError(
                f"k_scale and v_scale must have shape (1,), got {k_scale.shape} and {v_scale.shape}"
            )
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
        _validate_attention_dtype(q.dtype)
        cache_dtype = self._resolved_cache_dtype(q.dtype)
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if cache_dtype != q.dtype and cache_dtype != fp8_dtype:
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
        for name, tensor in [("k_scale", k_scale), ("v_scale", v_scale)]:
            if tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
            if (
                cache_dtype == fp8_dtype
                and not torch.all(torch.isfinite(tensor) & (tensor > 0)).item()
            ):
                raise ValueError(f"{name} must contain finite positive values")
        for name, tensor in [
            ("cu_seqlens_q", cu_seqlens_q),
            ("cache_seqlens", cache_seqlens),
            ("block_table", block_table),
        ]:
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
        if self.fuse_rope and max_total_len > self.max_position:
            raise ValueError(
                "cache_seqlens + q_len exceeds RoPE max_position: "
                f"max total length {max_total_len}, max_position {self.max_position}"
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

    def _get_rope_cos_sin(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_position is None:
            raise ValueError("max_position is required when fuse_rope=True")
        cached = self._rope_cos_cache.get((device, dtype))
        if cached is None:
            cached = base_freqs(
                self.rotary_dim,
                self.max_position,
                base=self.rope_base,
                dtype=dtype,
                device=device,
            )
            self._rope_cos_cache[(device, dtype)] = cached
        return cached

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
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
        )
        self.dtype = q.dtype
        call = self.attention_call(q.dtype)
        key = self.select_kernel_key(PAGED_PREFILL_KEYS, call)
        cos_table, sin_table = self._rope_tables(q.device, q.dtype)
        return self._get_kernel(key, call)(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            max_seqlen_q,
            cos_table,
            sin_table,
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
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}"
            )
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}"
            )
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.accum_dtype = accum_dtype

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype, max_seqlen_q: int) -> Kernel:
        def build() -> Kernel:
            return self.kernel_map["gqa_sliding_window_varlen_fwd_kernel"](
                batch=self.batch,
                heads=self.heads,
                heads_kv=self.heads_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_q,
                dim=self.dim,
                is_causal=self.is_causal,
                dtype=dtype,
                window_size_left=self.window_size_left,
                window_size_right=self.window_size_right,
                accum_dtype=self.accum_dtype,
                tune=self.tune,
            )

        # The launch bound is a constructor fact for this slot, so the
        # specialization carries it alongside the element type.
        return self.get_or_build_kernel(
            "gqa_sliding_window_varlen_fwd_kernel", key=(dtype, max_seqlen_q), build=build
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel = GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
        return {"gqa_sliding_window_varlen_fwd_kernel": kernel}

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
        for t, name in [(q, "q"), (k, "k"), (v, "v")]:
            if t.device.type != "cuda":
                raise ValueError(f"{name} must be on a cuda device, got {t.device}")
            if t.dtype != q.dtype:
                raise ValueError(f"{name} dtype {t.dtype} does not match q dtype {q.dtype}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if q.ndim != 3 or q.shape[1] != self.heads or q.shape[2] != self.dim:
            raise ValueError(
                f"q shape {q.shape} incompatible with heads={self.heads}, dim={self.dim}"
            )
        if k.ndim != 3 or k.shape[1] != self.heads_kv or k.shape[2] != self.dim:
            raise ValueError(
                f"k shape {k.shape} incompatible with heads_kv={self.heads_kv}, dim={self.dim}"
            )
        if v.ndim != 3 or v.shape[1] != self.heads_kv or v.shape[2] != self.dim:
            raise ValueError(
                f"v shape {v.shape} incompatible with heads_kv={self.heads_kv}, dim={self.dim}"
            )
        if cu_seqlens_q.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_q.shape[0] ({cu_seqlens_q.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})"
            )
        if cu_seqlens_k.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_k.shape[0] ({cu_seqlens_k.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})"
            )
        for cu, name in [(cu_seqlens_q, "cu_seqlens_q"), (cu_seqlens_k, "cu_seqlens_k")]:
            if cu.device.type != "cuda":
                raise ValueError(f"{name} must be on a cuda device, got {cu.device}")
            if cu.dtype != torch.int32:
                raise ValueError(f"{name} must have dtype int32, got {cu.dtype}")
            if not cu.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if cu_seqlens_q[0].item() != 0:
            raise ValueError(f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}")
        if cu_seqlens_k[0].item() != 0:
            raise ValueError(f"cu_seqlens_k[0] must be 0, got {cu_seqlens_k[0].item()}")
        if not torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if not torch.all(cu_seqlens_k[1:] >= cu_seqlens_k[:-1]):
            raise ValueError("cu_seqlens_k must be non-decreasing")
        if cu_seqlens_q[-1].item() > q.shape[0]:
            raise ValueError(
                f"cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()}) exceeds q.shape[0] ({q.shape[0]})"
            )
        if cu_seqlens_k[-1].item() > k.shape[0]:
            raise ValueError(
                f"cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()}) exceeds k.shape[0] ({k.shape[0]})"
            )
        actual_max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})"
            )

        self.dtype = q.dtype
        return self._get_kernel(q.dtype, max_seqlen_q).forward(q, k, v, cu_seqlens_q, cu_seqlens_k)

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time."
        )

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time."
        )
