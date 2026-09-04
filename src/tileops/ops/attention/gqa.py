import math
from typing import Callable, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    FlashAttnBwdPreprocessKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodePagedBs1Kernel,
    GQADecodePagedKernel,
    GQADenseCausalWsKernel,
    GQADenseSlidingWindowKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op
from ..rope import base_freqs
from .selection import PAGED_DECODE_KEYS, PAGED_PREFILL_KEYS, AttentionCall, fp8_dtype

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDenseFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
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


class GroupedQueryAttentionDenseFwdOp(Op):
    r"""Grouped-query attention over dense $Q$/$K$/$V$ tensors.

    By default the op computes causal attention,

    $$
    O = \operatorname{softmax}\!\left(
        \tfrac{1}{\sqrt D}\,QK^{\mathsf T} + M^{\mathrm{causal}}\right) V ,
    $$

    where $M^{\mathrm{causal}}$ is $0$ on visible entries and $-\infty$
    elsewhere. Accumulation is FP32 and the output takes the input dtype. FP8
    input is the one exception: it has no FP8 output, so ``dtype`` names a
    16-bit one.

    Every dimension is inferred from the ``forward`` tensors and none is fixed
    at construction, so one instance serves any shape. ``forward`` documents
    the shapes; these are the names it and the equations below use:

    | Dimension | Meaning |
    | --- | --- |
    | $B$ | Batch size |
    | $S_q$, $S_{kv}$ | Query and KV sequence length |
    | $H$, $H_{kv}$ | Query and KV head count |
    | $D$ | Head dimension |

    Dense means every batch entry shares one query length and one KV length,
    and its keys and values live in one contiguous tensor: neither ragged
    (varlen) batching nor a paged KV cache.

    Query heads are partitioned into $H_{kv}$ groups of $g = H / H_{kv}$, and
    the heads of one group share a KV head: writing $h$ for a query head and
    $r$ for a KV head, head $h$ attends to $r(h) = \lfloor h / g \rfloor$.
    $H$ must be divisible by $H_{kv}$.

    Each capability below is opt-in, enabled through a constructor parameter,
    the shape or dtype of the tensors passed to ``forward``, or both:

    | Capability | Enabled by |
    | --- | --- |
    | Attention without the causal mask | ``is_causal=False`` |
    | Sliding-window visibility | ``window_size_left``, ``window_size_right`` |
    | A custom score scale | ``sm_scale`` |
    | A logit softcap | ``softcap`` |
    | Rectangular attention, $S_q \ne S_{kv}$ | Nothing to set; read from the tensor shapes |
    | RoPE fused into $Q$ and $K$ | ``pos_encoding_mode="rope"``, plus ``rope_cos`` and ``rope_sin`` on the call |
    | FP8 $Q$/$K$/$V$, dequantized per KV head | ``float8_e4m3fn`` inputs, ``q_scale``/``k_scale``/``v_scale`` on the call, and ``dtype`` naming the 16-bit output |

    A call proceeds in the stages below, one per row of the table above. Using
    none of the optional capabilities leaves stages 2 and 3 doing nothing.
    Below, $i$ indexes a query row and $j$ a KV row; every quantity belongs to
    one batch entry, which the subscripts leave out.

    **1. Query positions.** Query row $i$ is bottom-right aligned with the KV
    sequence, so its position in the KV coordinate system is

    $$
    p_i = i + S_{kv} - S_q .
    $$

    Causal masking, windowing, and the query-side rotation all use $p_i$, not
    $i$. Causal and fused-RoPE calls therefore require $S_q \le S_{kv}$.

    **2. FP8 dequantization.** Each per-KV-head scale multiplies its tensor;
    for 16-bit inputs all three are implicitly one:

    $$
    \begin{aligned}
    \hat Q_{i,h} &= Q_{i,h} \cdot \mathrm{qscale}_{r(h)}, \\
    \hat K_{j,r} &= K_{j,r} \cdot \mathrm{kscale}_{r}, \\
    V'_{j,r} &= V_{j,r} \cdot \mathrm{vscale}_{r}.
    \end{aligned}
    $$

    **3. Fused rotation.** Let $R_x^{(d_r,\,\ell)}$ rotate the first
    ``rotary_dim`` dimensions at sequence position $x$ using ``rope_layout``
    $\ell$; without ``pos_encoding_mode="rope"`` it is the identity:

    $$
    Q'_{i,h} = R_{p_i}^{(d_r,\,\ell)}(\hat Q_{i,h}),
    \qquad
    K'_{j,r} = R_j^{(d_r,\,\ell)}(\hat K_{j,r}).
    $$

    **4. Scores.** With $\alpha$ = ``sm_scale``, default $1/\sqrt D$:

    $$
    Z_{h,i,j} = \alpha\,
        \langle Q'_{i,h}, K'_{j,r(h)} \rangle .
    $$

    **5. Visibility.** The causal flag and the window together decide which
    keys query row $i$ attends to; every other key contributes nothing to its
    output. With $w_L$ = ``window_size_left`` and $w_R$ =
    ``window_size_right``, each ``-1`` when unlimited, the visible set
    $\mathcal V_i$ holds key $j$ exactly when

    $$
    (\neg\mathrm{causal} \;\lor\; j \le p_i)
    \;\land\; (w_L=-1 \;\lor\; j \ge p_i-w_L)
    \;\land\; (w_R=-1 \;\lor\; j \le p_i+w_R).
    $$

    **6. Logits.** With $c$ = ``softcap``, capping and masking give

    $$
    L_{h,i,j} =
        \begin{cases}
        c\tanh(Z_{h,i,j}/c), & j\in\mathcal V_i \text{ and } c>0, \\
        Z_{h,i,j}, & j\in\mathcal V_i \text{ and } c=0, \\
        -\infty, & j\notin\mathcal V_i.
        \end{cases}
    $$

    **7. Output.** Normalizing over the KV axis and reducing the values gives

    $$
    \begin{aligned}
    P_{h,i,j} &=
        \operatorname{softmax}_{j}(L_{h,i,j}), \\
    O_{i,h} &= \sum_j P_{h,i,j}\,V'_{j,r(h)}.
    \end{aligned}
    $$
    """

    def __init__(
        self,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        pos_encoding_mode: str = "none",
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        dtype: Optional[torch.dtype] = None,
        *,
        target: Target = None,
    ) -> None:
        r"""Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Apply the causal mask, bottom-right aligned.
            window_size_left: Keys admitted left of the query row's KV
                position $p_i$; ``-1`` means unlimited.
            window_size_right: Keys admitted right of $p_i$; ``-1`` means
                unlimited.
            sm_scale: Score scale $\alpha$. ``None`` resolves to
                $1 / \sqrt{D}$ using the current call's head dimension.
            softcap: Positive cap $c$ replacing each raw score $z$ with
                $c \tanh(z / c)$. ``None`` or ``0`` disables capping.
            pos_encoding_mode: ``"none"``, or ``"rope"`` to fuse the rotary
                embedding into attention.
            rotary_dim: Rotated width of each head; even, at most $D$,
                default the full head dimension. Valid only with
                ``pos_encoding_mode="rope"``.
            rope_layout: ``"neox"`` (rotate split halves) or
                ``"interleaved"`` (rotate adjacent pairs).
            dtype: Output dtype. An FP8 call cannot return FP8 and the two
                16-bit types are equally valid, so ``float16`` or
                ``bfloat16`` must be named here; a 16-bit call has nothing to
                choose and accepts only ``None`` or its own input dtype.
            target: Backend target to serve this op, or ``None`` to decide
                from the input device.

        Raises:
            ValueError: A parameter is out of range, or the combination is
                inconsistent (e.g. ``rotary_dim`` without RoPE).
        """
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")
        if pos_encoding_mode not in ("none", "rope"):
            raise ValueError(f"pos_encoding_mode must be 'none' or 'rope', got {pos_encoding_mode}")
        if rotary_dim is not None and pos_encoding_mode != "rope":
            raise ValueError("rotary_dim requires pos_encoding_mode='rope'")
        if rotary_dim is not None:
            _validate_positive(rotary_dim=rotary_dim)
            if rotary_dim % 2 != 0:
                raise ValueError("rotary_dim must be even")
        if rope_layout not in ("neox", "interleaved"):
            raise ValueError("rope_layout must be 'neox' or 'interleaved'")
        if dtype is not None:
            _validate_attention_dtype(dtype)

        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        # A ``None`` scale depends on D, so it stays None here and resolves
        # when an implementation is requested for the current call.
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.dtype = dtype
        self.target = target
        self.dispatch_kernel()

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_dense": GQADenseCausalWsKernel,
            "gqa_dense_sliding_window": GQADenseSlidingWindowKernel,
        }

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

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> None:
        allowed = {torch.float16, torch.bfloat16, fp8_dtype()}
        if q.dtype not in allowed:
            raise ValueError("q must have float16, bfloat16, or float8_e4m3fn dtype")
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("q, k, and v must have the same dtype")
        is_fp8 = q.dtype == fp8_dtype()
        output_dtype = self.dtype or q.dtype
        if is_fp8 and self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP8 input requires dtype=torch.float16 or torch.bfloat16")
        if not is_fp8 and output_dtype != q.dtype:
            raise ValueError("16-bit output dtype must match q, k, and v")
        for name, scale in zip(
            ("q_scale", "k_scale", "v_scale"),
            (q_scale, k_scale, v_scale),
            strict=True,
        ):
            if scale is not None and scale.dtype != torch.float32:
                raise ValueError(f"{name} must have float32 dtype")
        for name, table in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
            if table is not None and table.dtype != output_dtype:
                raise ValueError(f"{name} must have dtype {output_dtype}")

    def eval_roofline(self) -> tuple[int, int]:
        """Keep this spec-only Op concrete until its roofline is implemented."""
        raise NotImplementedError("Dense GQA has no in-tree implementation yet")

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.ndim != 4:
                raise ValueError(f"{name} must be a rank-4 BSHD tensor")

        batch, seq_len_q, heads, dim = q.shape
        batch_kv, seq_len_kv, heads_kv, dim_kv = k.shape
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")
        if batch_kv != batch or dim_kv != dim:
            raise ValueError("q and k/v must have matching batch and head dimension")

        _validate_positive(batch=batch, seq_len_q=seq_len_q, seq_len_kv=seq_len_kv)
        _validate_gqa_dims(heads, heads_kv, dim)
        if self.is_causal and seq_len_q > seq_len_kv:
            raise ValueError("causal dense attention requires seq_len_q <= seq_len_kv")
        if self.pos_encoding_mode == "rope" and seq_len_q > seq_len_kv:
            raise ValueError("fused RoPE requires seq_len_q <= seq_len_kv")

        self._validate_dtypes(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)

        scales = (q_scale, k_scale, v_scale)
        has_scales = tuple(scale is not None for scale in scales)
        if any(has_scales) and not all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale must be supplied together")
        is_fp8 = q.dtype == fp8_dtype()
        if is_fp8 and not all(has_scales):
            raise ValueError("FP8 input requires q_scale, k_scale, and v_scale")
        if not is_fp8 and all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale are only valid for FP8 input")

        for name, tensor in (("k", k), ("v", v)):
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
        for name, scale in zip(("q_scale", "k_scale", "v_scale"), scales, strict=True):
            if scale is None:
                continue
            if scale.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if tuple(scale.shape) != (batch, heads_kv):
                raise ValueError(f"{name} must have shape {(batch, heads_kv)}")

        if (rope_cos is None) != (rope_sin is None):
            raise ValueError("rope_cos and rope_sin must be supplied together")
        if self.pos_encoding_mode != "rope":
            if rope_cos is not None:
                raise ValueError("RoPE tables require pos_encoding_mode='rope'")
            return
        if rope_cos is None or rope_sin is None:
            raise ValueError("pos_encoding_mode='rope' requires rope_cos and rope_sin")

        expected_columns = _rope_rotary_dim(dim, self.rotary_dim) // 2
        for name, table in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
            if table.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if table.ndim != 2:
                raise ValueError(f"{name} must be 2-dimensional")
            if table.shape[0] < seq_len_kv or table.shape[1] != expected_columns:
                raise ValueError(
                    f"{name} must have shape [max_position >= {seq_len_kv}, {expected_columns}]"
                )
        if rope_cos.shape != rope_sin.shape:
            raise ValueError("rope_cos and rope_sin must have the same shape")

    def _validate_builtin_call(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Reject features not implemented by the in-tree Dense kernels."""
        if not self.is_causal:
            raise ValueError("Dense GQA currently supports causal attention only")
        if q.shape[-1] != 128:
            raise ValueError("Dense GQA currently requires head dimension 128")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Dense GQA currently supports float16 and bfloat16 inputs only")
        uses_window = self.window_size_left != -1 or self.window_size_right != -1
        if uses_window and q.shape[1] != k.shape[1]:
            raise ValueError("Dense sliding-window GQA currently requires equal Q and KV lengths")

    @staticmethod
    def _canonicalize_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Return contiguous tensors in manifest order, preserving None slots."""
        return tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)
        )

    def _get_kernel(
        self, inputs: tuple[Optional[torch.Tensor], ...]
    ) -> Callable[..., torch.Tensor]:
        """Resolve the implementation stored in the Op's single cache layer."""
        q, k, _v, _q_scale, _k_scale, _v_scale, rope_cos, _rope_sin = inputs
        assert q is not None and k is not None
        batch, seq_len_q, heads, dim = q.shape
        _, seq_len_kv, heads_kv, _ = k.shape
        uses_window = self.window_size_left != -1 or self.window_size_right != -1
        role = "gqa_dense_sliding_window" if uses_window else "gqa_dense"
        rope_on = self.pos_encoding_mode == "rope"
        rope_kwargs = {
            "fuse_rope": rope_on,
            "max_position": rope_cos.shape[0] if rope_cos is not None else 1,
            "rotary_dim": _rope_rotary_dim(dim, self.rotary_dim) if rope_on else 0,
            "rope_layout": self.rope_layout,
        }

        def build() -> Kernel:
            self._validate_builtin_call(q, k)
            if uses_window:
                return self.kernel_map[role](
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    seq_len=seq_len_q,
                    dim=dim,
                    is_causal=self.is_causal,
                    window_size_left=self.window_size_left,
                    window_size_right=self.window_size_right,
                    dtype=q.dtype,
                    sm_scale=self.sm_scale,
                    softcap=self.softcap,
                    **rope_kwargs,
                    device_index=q.device.index,
                )
            return self.kernel_map[role](
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                dim=dim,
                dtype=q.dtype,
                sm_scale=self.sm_scale,
                softcap=self.softcap,
                **rope_kwargs,
                device_index=q.device.index,
            )

        if uses_window or rope_on:
            # Sliding and RoPE still compile exact sequence lengths. The plain
            # causal WS kernel accepts its sequence extents at runtime.
            key = (
                q.dtype,
                tuple(q.shape),
                tuple(k.shape),
                None if rope_cos is None else tuple(rope_cos.shape),
            )
        else:
            key = (
                q.dtype,
                batch,
                heads,
                heads_kv,
                dim,
            )
        return self.get_or_build_kernel(role, inputs, key=key, build=build)

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
        r"""Run dense GQA attention over one set of $Q$/$K$/$V$ tensors.

        $Q$, $K$, and $V$ are laid out row-major in BSHD axis order, so the
        head dimension is the contiguous one. Every input must be on the same
        device as ``q``; a non-contiguous input is copied to a contiguous one
        before the kernel runs.

        Args:
            q: Queries, $[B \times S_q \times H \times D]$; ``float16``,
                ``bfloat16``, or ``float8_e4m3fn``.
            k: Keys, $[B \times S_{kv} \times H_{kv} \times D]$, same dtype
                as ``q``.
            v: Values, $[B \times S_{kv} \times H_{kv} \times D]$, same
                dtype as ``q``.
            q_scale: FP8 dequantization scales for ``q``, one per batch and
                KV head, $[B \times H_{kv}]$, ``float32``. The three scales
                are required together for FP8 input and invalid otherwise.
            k_scale: Scales for ``k``, one per batch and KV head,
                $[B \times H_{kv}]$, ``float32``.
            v_scale: Scales for ``v``, one per batch and KV head,
                $[B \times H_{kv}]$, ``float32``.
            rope_cos: RoPE cosine table indexed by KV position and rotated
                pair, $[P \times d_r / 2]$ with $P \ge S_{kv}$ and $d_r$ =
                ``rotary_dim``, in the output dtype. The two tables are
                required together with ``pos_encoding_mode="rope"`` and
                invalid otherwise.
            rope_sin: RoPE sine table, same layout, shape, and dtype as
                ``rope_cos``.

        Returns:
            Attention output, $[B \times S_q \times H \times D]$, laid out
            like ``q`` and contiguous. Its dtype is ``dtype`` for FP8 input,
            and the input dtype otherwise.

        Raises:
            ValueError: Shapes, dtypes, devices, or optional-input
                combinations violate the contract above.
        """
        self._validate_forward_inputs(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)
        inputs = self._canonicalize_inputs(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)
        kernel = self._get_kernel(inputs)
        return kernel(*inputs)


class GroupedQueryAttentionPrefillVarlenFwdOp(Op):
    """Packed variable-length GQA prefill. Layout: THD.

    ``cu_seqlens_q`` and ``cu_seqlens_kv`` describe packed per-request ranges.
    Causal prefill uses bottom-right alignment for each request independently:
    key position ``j`` is visible to query position ``i`` iff
    ``j <= i + (kv_len - q_len)``.
    """

    def __init__(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            max_seqlen_q: Longest request in the packed q the op will be called with.
            max_seqlen_kv: Longest request in the packed kv.
            is_causal: Whether a query may attend past its own position.
            sm_scale: Softmax scale; ``None`` takes ``dim ** -0.5`` from the call.
            softcap: Score softcap, or ``None``.
            validate_inputs: Whether to read the offsets back and check them.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        _validate_positive(max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_causal = is_causal
        # Resolved in `forward`: the default is `dim ** -0.5`, and `dim` comes from the call.
        self._sm_scale_arg = sm_scale
        self.softcap = _score_softcap(softcap)
        self.validate_inputs = validate_inputs
        self._roofline_kwargs = None

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_kv_shape: tuple[int, ...],
    ) -> Dict[str, tuple[int, ...]]:
        return {"o": tuple(q_shape)}

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
        _validate_attention_dtype(dtype)
        key = (self.batch, self.heads, self.heads_kv, self.dim, self.sm_scale, dtype)

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

        return self.get_or_build_kernel(
            "gqa_prefill_varlen_fwd_kernel", inputs, key=key, build=build
        )

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

        if q.ndim != 3:
            raise ValueError(f"Expected q shape [T, H, D], got {tuple(q.shape)}")
        heads, dim = q.shape[1], q.shape[2]
        for name in ("k", "v"):
            tensor = tensors[name]
            if tensor.ndim != 3 or tensor.shape[2] != dim:
                raise ValueError(
                    f"Expected {name} shape [T, H_kv, {dim}], got {tuple(tensor.shape)}"
                )
            if tensor.dtype != q.dtype:
                raise ValueError(f"Expected {name}.dtype {q.dtype}, got {tensor.dtype}")
        if k.shape[1] != v.shape[1]:
            raise ValueError(f"k and v must share H_kv; got {k.shape[1]} and {v.shape[1]}")
        _validate_gqa_dims(heads, k.shape[1], dim)

        # The batch size is read off this shape, so its rank is checked before that read.
        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] < 2:
            raise ValueError(
                f"cu_seqlens_q must be a 1D tensor of at least two bounds; "
                f"got {tuple(cu_seqlens_q.shape)}"
            )
        batch = cu_seqlens_q.shape[0] - 1
        for name in ("cu_seqlens_q", "cu_seqlens_kv"):
            tensor = tensors[name]
            if tuple(tensor.shape) != (batch + 1,):
                raise ValueError(f"Expected {name} shape {(batch + 1,)}, got {tuple(tensor.shape)}")
            if tensor.dtype != torch.int32:
                raise ValueError(f"Expected {name}.dtype torch.int32, got {tensor.dtype}")
        _validate_positive(batch=batch)

        self.batch, self.heads, self.heads_kv, self.dim = batch, heads, k.shape[1], dim
        self.sm_scale = _attention_scale(dim, self._sm_scale_arg)

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
        """Run the op on ``q``, ``k``, ``v``, ``cu_seqlens_q`` and ``cu_seqlens_kv``."""
        self._validate_forward_inputs(q, k, v, cu_seqlens_q, cu_seqlens_kv)
        self.dtype = q.dtype
        tensors = (q, k, v, cu_seqlens_q, cu_seqlens_kv)
        output = self._get_kernel(tensors, q.dtype)(*tensors)
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

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


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
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            max_pages_per_req: Manifest ``params.max_pages_per_req``, ``int``.
            page_size: Manifest ``params.page_size``, ``int``.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            cache_dtype: Manifest ``params.cache_dtype``, ``dtype | None``, default ``None``.
            sm_scale: Manifest ``params.sm_scale``, ``float | None``, default ``None``.
            softcap: Manifest ``params.softcap``, ``float | None``, default ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
            fuse_rope: Manifest ``params.fuse_rope``, ``bool``, default ``False``.
            rope_base: Manifest ``params.rope_base``, ``float``, default ``10000.0``.
            max_position: Manifest ``params.max_position``, ``int | None``, default ``None``.
            rotary_dim: Manifest ``params.rotary_dim``, ``int | None``, default ``None``.
        """
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

    def _get_kernel(
        self, inputs: "tuple[torch.Tensor | None, ...]", key: str, call: AttentionCall
    ) -> Kernel:
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

        return self.get_or_build_kernel(key, inputs, key=call.dtype, build=build)

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

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_new_shape: tuple[int, ...],
        v_new_shape: tuple[int, ...],
        k_pages_shape: tuple[int, ...],
        v_pages_shape: tuple[int, ...],
        k_scale_shape: tuple[int, ...],
        v_scale_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cache_seqlens_shape: tuple[int, ...],
        block_table_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o.shape == q.shape``."""
        return {"o": tuple(q_shape)}

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
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k_new: Input tensor, dtype ``same_as(q)``.
            v_new: Input tensor, dtype ``same_as(q)``.
            k_pages: Input tensor, dtype ``float16 | bfloat16 | float8_e4m3fn``.
            v_pages: Input tensor, dtype ``same_as(k_pages)``.
            k_scale: Input tensor, dtype ``float32``.
            v_scale: Input tensor, dtype ``float32``.
            cu_seqlens_q: Input tensor, dtype ``int32``.
            cache_seqlens: Input tensor, dtype ``int32``.
            block_table: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (total_q, H, D)``.
        """
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
        return self._get_kernel(
            (
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
            ),
            key,
            call,
        )(
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

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


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
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernels(
        self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype
    ) -> tuple[Kernel, Kernel]:
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
                "gqa_bwd_preprocess_kernel", inputs, key=dtype, build=build_preprocess
            ),
            self.get_or_build_kernel("gqa_bwd_kernel", inputs, key=dtype, build=build_backward),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel": GQABwdWgmmaPipelinedKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        o_shape: tuple[int, ...],
        do_shape: tuple[int, ...],
        lse_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: each gradient has the shape of what it is for."""
        return {"dq": tuple(q_shape), "dk": tuple(k_shape), "dv": tuple(v_shape)}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            q: Input tensor, dtype ``float16 | bfloat16``.
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            o: Input tensor, dtype ``same_as(q)``.
            do: Input tensor, dtype ``same_as(q)``.
            lse: Input tensor, dtype ``float32``.

        Returns:
            ``dq``, ``dk``, ``dv``, as the manifest declares. Shape rules: ``dq.shape == (B, S, H, D)``; ``dk.shape == (B, S, H_kv, D)``; ``dv.shape == (B, S, H_kv, D)``.
        """
        do = do.contiguous()
        self._validate_dtypes(q, k, v, o, do, lse)
        self.dtype = q.dtype
        prep_kernel, kernel = self._get_kernels((q, k, v, o, do, lse), q.dtype)
        delta = prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(q.dtype)
        dk, dv = dk.to(q.dtype), dv.to(q.dtype)
        return dq, dk, dv

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged GQA decode with dynamic KV cache. Layout: ``Q`` $[batch \\times heads \\times dim]$ (BHD);
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
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            sm_scale: Manifest ``params.sm_scale``, ``float | None``, default ``None``.
            softcap: Manifest ``params.softcap``, ``float | None``, default ``None``.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
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

    def _get_kernel(self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype) -> Kernel:
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

        return self.get_or_build_kernel(key, inputs, key=dtype, build=build)

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

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        real_seqlen_kv_shape: tuple[int, ...],
        block_table_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o.shape == q.shape``."""
        return {"o": tuple(q_shape)}

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
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            real_seqlen_kv: Input tensor, dtype ``int32``.
            block_table: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, H, D)``.
        """
        self.dtype = q.dtype
        return self._get_kernel((q, k, v, real_seqlen_kv, block_table), q.dtype)(
            q, k, v, real_seqlen_kv, block_table
        )

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)


class GroupedQueryAttentionSlidingWindowVarlenFwdOp(Op):
    """Variable-length GQA forward with sliding window attention.

    Inputs are packed (no padding); per-sample boundaries are given via
    cu_seqlens arrays.  seqlen_q and seqlen_k may differ per sample:

      offset = seqlen_k - seqlen_q  (per sample, FA3 bottom-right convention)

    A token at local q_pos attends to local k_pos when ALL conditions hold:
      k_pos <= q_pos + offset                      (is_causal=True)
      k_pos >= q_pos + offset - window_size_left   (window_size_left >= 0)
      k_pos <= q_pos + offset + window_size_right  (window_size_right >= 0)

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
        """Build the op. Shapes and dtype are taken from the first call.

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

    def _get_kernel(
        self, inputs: "tuple[torch.Tensor | None, ...]", dtype: torch.dtype, max_seqlen_q: int
    ) -> Kernel:
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
            "gqa_sliding_window_varlen_fwd_kernel", inputs, key=(dtype, max_seqlen_q), build=build
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel = GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
        return {"gqa_sliding_window_varlen_fwd_kernel": kernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_k_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o.shape == q.shape``."""
        return {"o": tuple(q_shape)}

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
            q: Query tensor, shape $[total\\_q \\times heads \\times dim]$.
            k: Key tensor, shape $[total\\_k \\times heads\\_kv \\times dim]$.
            v: Value tensor, shape $[total\\_k \\times heads\\_kv \\times dim]$.
            cu_seqlens_q: Cumulative Q lengths, shape $[batch+1]$, dtype int32.
            cu_seqlens_k: Cumulative K lengths, shape $[batch+1]$, dtype int32.
            max_seqlen_q: Maximum Q sequence length across the batch.

        Returns:
            Output tensor, shape $[total\\_q \\times heads \\times dim]$.
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
        return self._get_kernel(
            (q, k, v, cu_seqlens_q, cu_seqlens_k), q.dtype, max_seqlen_q
        ).forward(q, k, v, cu_seqlens_q, cu_seqlens_k)

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

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
