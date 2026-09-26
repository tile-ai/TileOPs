import dataclasses
import math
from typing import Callable, ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    FlashAttnBwdPreprocessKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADecodePagedBs1Kernel,
    GQADecodePagedKernel,
    GQADenseFP8DecodeKernel,
    GQADenseFP8Kernel,
    GQADenseSlidingWindowKernel,
    GQADenseWsKernel,
    GQAPrefillPagedWithFP8KVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheFwdKernel,
    GQAPrefillPagedWithKVCacheRopeFwdKernel,
    GQAPrefillVarlenFwdKernel,
    GQAPrefillVarlenWSFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op
from ..rope import base_freqs
from .selection import AttentionCall, fp8_dtype

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDenseFwdOp",
    "GroupedQueryAttentionPagedFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionPrefillVarlenFwdOp",
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp",
    "GroupedQueryAttentionVarlenFwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
]


def _validate_attention_dtype(dtype: torch.dtype) -> None:
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Expected dtype torch.float16 or torch.bfloat16, got {dtype}")


def _dense_decode_split_capacity(seq_len_kv: int) -> int:
    """Bucket a runtime KV extent by the largest feasible split tier."""
    full_tiles = max(1, seq_len_kv // 64)
    return min(32, 1 << (full_tiles.bit_length() - 1))


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
    input is the one exception: it has no FP8 output, so ``out_dtype`` names a
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
    | FP8 $Q$/$K$/$V$, dequantized per KV head | ``float8_e4m3fn`` inputs, ``q_scale``/``k_scale``/``v_scale`` on the call, and ``out_dtype`` naming the 16-bit output |

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

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_dense": GQADenseWsKernel,
        "gqa_dense_decode": GQADecodeKernel,
        "gqa_dense_decode_bs1": GQADecodeBs1Kernel,
        "gqa_dense_fp8": GQADenseFP8Kernel,
        "gqa_dense_fp8_decode": GQADenseFP8DecodeKernel,
        "gqa_dense_decode_long_context": GQADecodeLongContextKernel,
        "gqa_dense_sliding_window": GQADenseSlidingWindowKernel,
    }

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
        out_dtype: Optional[torch.dtype] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            out_dtype: Manifest ``params.out_dtype``. An FP8 call cannot return FP8 and the two
                16-bit types are equally valid, so ``float16`` or
                ``bfloat16`` must be named here; a 16-bit call has nothing to
                choose and accepts only ``None`` or its own input dtype.
            target: Backend target to serve this op, or ``None`` to decide
                from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Whether to autotune, applied when a kernel is first built.

        Raises:
            ValueError: ``sm_scale`` is not finite or ``softcap`` is negative.
        """
        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")
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
        self.out_dtype = out_dtype
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """Dense attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def _validate_builtin_call(self, q: torch.Tensor, k: torch.Tensor) -> None:
        """Reject features not implemented by the in-tree Dense kernels."""
        if q.shape[-1] != 128:
            raise ValueError("Dense GQA currently requires head dimension 128")
        uses_window = self.window_size_left != -1 or self.window_size_right != -1
        if uses_window and q.shape[1] != k.shape[1]:
            raise ValueError("Dense sliding-window GQA currently requires equal Q and KV lengths")

    def dense_call(self, inputs: tuple[Optional[torch.Tensor], ...]) -> AttentionCall:
        """State what one contiguous call is, for selection to filter against."""
        q, k, _v, _q_scale, _k_scale, _v_scale, rope_cos, _rope_sin = inputs
        assert q is not None and k is not None
        batch, seq_len_q, heads, dim = q.shape
        _, seq_len_kv, heads_kv, _ = k.shape
        rope_on = self.pos_encoding_mode == "rope"
        is_fp8 = q.dtype == fp8_dtype()
        return AttentionCall(
            # The element type an implementation is compiled for. FP8 inputs are
            # computed in the type the op was constructed with.
            dtype=self.out_dtype if is_fp8 else q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_seqlen_q=seq_len_q,
            seqlen_kv=seq_len_kv,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=is_fp8,
            fuse_rope=rope_on,
            max_position=rope_cos.shape[0] if rope_cos is not None else 1,
            rotary_dim=_rope_rotary_dim(dim, self.rotary_dim) if rope_on else 0,
            rope_layout=self.rope_layout,
            tune=self.tune,
            device=q.device,
        )

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
            like ``q`` and contiguous. Its dtype is ``out_dtype`` for FP8 input,
            and the input dtype otherwise.

        Raises:
            ValueError: Shapes, dtypes, devices, or optional-input
                combinations violate the contract above, or the in-tree kernels do
                not implement the call (head dimension other than 128, or a window
                over unequal Q and KV lengths).
        """
        return self._call_boundary(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)

    def _eager_forward(
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
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._validate_builtin_call(q, k)
        inputs = tuple(
            tensor.contiguous() if tensor is not None else None
            for tensor in (q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin)
        )
        return self.kernel_for("gqa_dense", inputs, self.dense_call(inputs))(*inputs)


class GroupedQueryAttentionVarlenFwdOp(Op):
    """Grouped-query attention over packed THD tensors.

    ``cu_seqlens_q`` and ``cu_seqlens_kv`` delimit each request. The interface
    covers both prefill and decode; tensor geometry and sequence metadata come
    from each call, while mask, score, out_dtype, and RoPE semantics are fixed at
    construction. The current BUILTIN path implements 16-bit regular and
    sliding-window attention; FP8 and fused RoPE remain part of the public
    contract for later kernel migrations.
    """

    compile_boundary = True

    def __init__(
        self,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        out_dtype: Optional[torch.dtype] = None,
        pos_encoding_mode: str = "none",
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        validate_inputs: bool = False,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        target: Target = None,
        tune: bool = False,
    ) -> None:
        """Configure packed variable-length GQA semantics.

        Args:
            is_causal: Apply a bottom-right-aligned causal mask per request.
            window_size_left: Visible keys to the left; ``-1`` is unlimited.
            window_size_right: Visible keys to the right; ``-1`` is unlimited.
            sm_scale: Score scale, or ``None`` for ``1 / sqrt(head_dim)``.
            softcap: Positive score cap; ``None`` or zero disables it.
            out_dtype: Output dtype, inferred from the input when omitted.
            pos_encoding_mode: ``"none"`` or ``"rope"``.
            rotary_dim: Even rotated width; ``None`` uses the full head dimension.
            rope_layout: ``"neox"`` or ``"interleaved"``.
            validate_inputs: Check cumulative offsets against packed tensors on the CPU.
            kernel_map: Optional in-tree kernel overrides.
            target: Backend target, or ``None`` to resolve from the input device.
            tune: Autotune a kernel when it is first built.
        """
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")
        if pos_encoding_mode not in ("none", "rope"):
            raise ValueError("pos_encoding_mode must be 'none' or 'rope'")
        if rotary_dim is not None and pos_encoding_mode != "rope":
            raise ValueError("rotary_dim requires pos_encoding_mode='rope'")
        if rotary_dim is not None and (rotary_dim <= 0 or rotary_dim % 2):
            raise ValueError("rotary_dim must be a positive even integer")
        if rope_layout not in ("neox", "interleaved"):
            raise ValueError("rope_layout must be 'neox' or 'interleaved'")
        if out_dtype is not None:
            _validate_attention_dtype(out_dtype)
        resolved_softcap = _score_softcap(softcap)
        if (window_size_left != -1 or window_size_right != -1) and (
            sm_scale is not None or resolved_softcap != 0.0
        ):
            raise ValueError("windowed Varlen GQA does not yet support sm_scale or softcap")

        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = resolved_softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.validate_inputs = validate_inputs
        self.target = target
        self._roofline_kwargs: Optional[dict] = None
        self._last_input_dtype: Optional[torch.dtype] = None
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_varlen": GQAPrefillVarlenFwdKernel,
            "gqa_varlen_ws": GQAPrefillVarlenWSFwdKernel,
            "gqa_varlen_sliding_window": GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
        cu_seqlens_kv_shape: tuple[int, ...],
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
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
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
        if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_kv.dtype != torch.int32:
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must have int32 dtype")
        for name, tensor in (
            ("q_scale", q_scale),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
        ):
            if tensor is not None and tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have float32 dtype")
        for name, tensor in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
            if tensor is not None and tensor.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"{name} must have float16 or bfloat16 dtype")

    def eval_roofline(self) -> tuple[int, int]:
        if self._roofline_kwargs is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() call"
            )
        from tileops.perf.formulas import visible_scores

        call = self._roofline_kwargs
        q_bounds = call["cu_seqlens_q"].tolist()
        kv_bounds = call["cu_seqlens_kv"].tolist()
        visible = sum(
            visible_scores(
                q_end - q_start,
                kv_end - kv_start,
                call["is_causal"],
                call["window_size_left"],
                call["window_size_right"],
            )
            for q_start, q_end, kv_start, kv_end in zip(
                q_bounds, q_bounds[1:], kv_bounds, kv_bounds[1:], strict=False
            )
        )
        heads, heads_kv, dim = call["heads"], call["heads_kv"], call["dim"]
        elem = call["dtype"].itemsize
        # q, k, v and o once each, and the two cumulative-length tensors.
        moved = (2 * call["total_q"] * heads + 2 * call["total_k"] * heads_kv) * dim * elem
        return 4 * heads * visible * dim, moved + 2 * (call["batch"] + 1) * 4

    def compute_roof(self) -> str:
        """Varlen attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self._last_input_dtype)

    def varlen_call(self, inputs: tuple[Optional[torch.Tensor], ...]) -> AttentionCall:
        """Describe one packed call using tensor shapes and Op semantics."""
        q, k, _v, cu_q, _cu_kv, _qs, _ks, _vs, rope_cos, _rope_sin = inputs
        assert q is not None and k is not None and cu_q is not None
        _, heads, dim = q.shape
        _, heads_kv, _ = k.shape
        return AttentionCall(
            dtype=self.out_dtype or q.dtype,
            batch=cu_q.shape[0] - 1,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=q.dtype == fp8_dtype(),
            is_uniform=False,
            empty_kv=k.shape[0] == 0,
            fuse_rope=self.pos_encoding_mode == "rope",
            max_position=rope_cos.shape[0] if rope_cos is not None else 1,
            rotary_dim=_rope_rotary_dim(dim, self.rotary_dim)
            if self.pos_encoding_mode == "rope"
            else 0,
            rope_layout=self.rope_layout,
            tune=self.tune,
            device=q.device,
        )

    def _get_kernel(
        self, inputs: tuple[Optional[torch.Tensor], ...]
    ) -> Callable[..., torch.Tensor]:
        """Resolve the implementation stored in the Op's single cache layer."""
        return self.kernel_for("gqa_varlen", inputs, self.varlen_call(inputs))

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> None:
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.ndim != 3:
                raise ValueError(f"{name} must be a rank-3 THD tensor")
        if k.shape != v.shape:
            raise ValueError("k and v must have the same shape")

        _, heads, dim = q.shape
        _, heads_kv, dim_kv = k.shape
        if dim_kv != dim:
            raise ValueError("q and k/v must have the same head dimension")
        _validate_gqa_dims(heads, heads_kv, dim)

        if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] < 2:
            raise ValueError("cu_seqlens_q must be rank 1 with at least two entries")
        if cu_seqlens_kv.shape != cu_seqlens_q.shape:
            raise ValueError("cu_seqlens_q and cu_seqlens_kv must have the same shape")
        batch = cu_seqlens_q.shape[0] - 1

        GroupedQueryAttentionVarlenFwdOp._validate_dtypes(
            self,
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

        output_dtype = self.out_dtype or q.dtype
        is_fp8 = q.dtype == fp8_dtype()
        if is_fp8 and self.out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP8 input requires a 16-bit output dtype")
        if not is_fp8 and output_dtype != q.dtype:
            raise ValueError("16-bit output dtype must match q, k, and v")

        scales = (q_scale, k_scale, v_scale)
        has_scales = tuple(scale is not None for scale in scales)
        if any(has_scales) and not all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale must be supplied together")
        if is_fp8 and not all(has_scales):
            raise ValueError("FP8 input requires q_scale, k_scale, and v_scale")
        if not is_fp8 and all(has_scales):
            raise ValueError("q_scale, k_scale, and v_scale are only valid for FP8 input")

        tensors = (
            ("k", k),
            ("v", v),
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_kv", cu_seqlens_kv),
        )
        for name, tensor in tensors:
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
        for name, scale in zip(("q_scale", "k_scale", "v_scale"), scales, strict=True):
            if scale is None:
                continue
            if scale.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
            if tuple(scale.shape) != (batch, heads_kv):
                raise ValueError(f"{name} must have shape {(batch, heads_kv)}")

        if self.validate_inputs:
            for name, offsets, total in (
                ("cu_seqlens_q", cu_seqlens_q, q.shape[0]),
                ("cu_seqlens_kv", cu_seqlens_kv, k.shape[0]),
            ):
                bounds = [int(value) for value in offsets.detach().cpu().tolist()]
                if bounds[0] != 0:
                    raise ValueError(f"{name}[0] must equal 0")
                if bounds[-1] != total:
                    raise ValueError(f"{name}[-1] must equal {total}")
                if any(end < start for start, end in zip(bounds[:-1], bounds[1:], strict=True)):
                    raise ValueError(f"{name} must be non-decreasing")

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
            if table.dtype != output_dtype:
                raise ValueError(f"{name} must have dtype {output_dtype}")
            if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != expected_columns:
                raise ValueError(f"{name} must have shape [max_position, {expected_columns}]")
        if rope_cos.shape != rope_sin.shape:
            raise ValueError("rope_cos and rope_sin must have the same shape")

    @staticmethod
    def _canonicalize_inputs(
        *inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        return tuple(tensor.contiguous() if tensor is not None else None for tensor in inputs)

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
        """Run packed Varlen GQA; Q/K/V use ``[total_tokens, heads, dim]``."""
        return self._call_boundary(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale, rope_cos, rope_sin
        )

    def _eager_forward(
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
        """Validate, resolve the implementation and launch it."""
        self._validate_forward_inputs(
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
        inputs = self._canonicalize_inputs(
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
        kernel = self._get_kernel(inputs)
        output = kernel(*inputs)
        self._last_input_dtype = q.dtype
        self._roofline_kwargs = {
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch": cu_seqlens_q.shape[0] - 1,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_kv": cu_seqlens_kv,
            "total_q": q.shape[0],
            "total_k": k.shape[0],
            "is_causal": self.is_causal,
            "window_size_left": self.window_size_left,
            "window_size_right": self.window_size_right,
            "heads": q.shape[1],
            "heads_kv": k.shape[1],
            "dim": q.shape[2],
            "dtype": q.dtype,
        }
        return output


class GroupedQueryAttentionPrefillVarlenFwdOp(GroupedQueryAttentionVarlenFwdOp):
    """Compatibility API retained while the unified Varlen contract is spec-only.

    With ``validate_inputs=True`` a call also checks, on the CPU, that the offsets span
    the packed tensors, that every request is non-empty and within ``max_seqlen_*``, and
    that a causal request has no more queries than keys.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_prefill_varlen_fwd_kernel": GQAPrefillVarlenFwdKernel
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The scores this call's request lengths make visible, which its flops follow."""
        from tileops.perf.formulas import packed_visible_scores

        return {"visible_scores": packed_visible_scores(self.last_call, "cu_seqlens_kv")}

    def __init__(
        self,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        validate_inputs: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure the legacy regular-Varlen implementation.

        This compatibility API remains available until its FP8 and RoPE
        capabilities have migrated to `GroupedQueryAttentionVarlenFwdOp`.

        Args:
            max_seqlen_q: Longest query request a call may carry.
            max_seqlen_kv: Longest key request a call may carry.
            is_causal: Apply a bottom-right-aligned causal mask per request.
            sm_scale: Score scale, or ``None`` for ``1 / sqrt(head_dim)``.
            softcap: Positive score cap; ``None`` or zero disables it.
            validate_inputs: Check the offsets against the packed tensors on the CPU.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = -1
        self.window_size_right = -1
        self.out_dtype = None
        self.pos_encoding_mode = "none"
        self.rotary_dim = None
        self.rope_layout = "neox"
        self.validate_inputs = validate_inputs
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """Varlen attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def _check_offsets(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
        cu_q = [int(value) for value in cu_seqlens_q.detach().cpu().tolist()]
        cu_kv = [int(value) for value in cu_seqlens_kv.detach().cpu().tolist()]
        if cu_q[0] != 0 or cu_q[-1] != q.shape[0]:
            raise ValueError("cu_seqlens_q must span the packed q tensor")
        if cu_kv[0] != 0 or cu_kv[-1] != k.shape[0]:
            raise ValueError("cu_seqlens_kv must span the packed k/v tensors")
        q_lens = [end - start for start, end in zip(cu_q[:-1], cu_q[1:], strict=True)]
        kv_lens = [end - start for start, end in zip(cu_kv[:-1], cu_kv[1:], strict=True)]
        if any(length <= 0 for length in q_lens):
            raise ValueError("all q sequence lengths must be positive")
        if any(length <= 0 for length in kv_lens):
            raise ValueError("all kv sequence lengths must be positive")
        if max(q_lens) > self.max_seqlen_q:
            raise ValueError("max_seqlen_q is smaller than an actual q sequence")
        if max(kv_lens) > self.max_seqlen_kv:
            raise ValueError("max_seqlen_kv is smaller than an actual kv sequence")
        if self.is_causal and any(
            q_len > kv_len for q_len, kv_len in zip(q_lens, kv_lens, strict=True)
        ):
            raise ValueError("causal varlen prefill requires every q_len <= kv_len")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Run regular attention over packed variable-length Q, K, and V."""
        return self._call_boundary(q, k, v, cu_seqlens_q, cu_seqlens_kv)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        if self.validate_inputs:
            self._check_offsets(q, k, cu_seqlens_q, cu_seqlens_kv)
        inputs = self._canonicalize_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, None, None, None, None, None
        )
        kernel = self.kernel_for("gqa_prefill_varlen_fwd_kernel", inputs, self.varlen_call(inputs))
        return kernel(*inputs)


class GroupedQueryAttentionSlidingWindowVarlenFwdOp(GroupedQueryAttentionVarlenFwdOp):
    """Compatibility sliding-window API retained during the unified migration."""

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_sliding_window_varlen_fwd_kernel": GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
    }

    # Kernel configuration, not contract: the accumulator dtype.
    accum_dtype: ClassVar[torch.dtype] = torch.float32

    def __init__(
        self,
        max_seqlen_q: int,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure the legacy sliding-window Varlen implementation.

        Args:
            max_seqlen_q: Longest query request a call may carry.
            is_causal: Apply a bottom-right-aligned causal mask per request.
            window_size_left: Visible keys to the left; ``-1`` is unlimited.
            window_size_right: Visible keys to the right; ``-1`` is unlimited.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.max_seqlen_q = max_seqlen_q
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.sm_scale = None
        self.softcap = 0.0
        self.out_dtype = None
        self.pos_encoding_mode = "none"
        self.rotary_dim = None
        self.rope_layout = "neox"
        self.validate_inputs = False
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def roofline_inputs(self) -> "dict[str, int]":
        """The scores this call's request lengths and window make visible."""
        from tileops.perf.formulas import packed_visible_scores

        return {"visible_scores": packed_visible_scores(self.last_call, "cu_seqlens_k")}

    def compute_roof(self) -> str:
        """Varlen attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """Run sliding-window attention over packed variable-length inputs."""
        return self._call_boundary(q, k, v, cu_seqlens_q, cu_seqlens_k)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        inputs = self._canonicalize_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_k, None, None, None, None, None
        )
        call = dataclasses.replace(self.varlen_call(inputs), accum_dtype=self.accum_dtype)
        kernel = self.kernel_for("gqa_sliding_window_varlen_fwd_kernel", inputs, call)
        return kernel(*inputs)


class GroupedQueryAttentionPagedFwdOp(Op):
    """Grouped-query attention over a caller-owned paged KV cache.

    Packed Q and its cumulative sequence lengths cover both prefill and decode.
    ``page_table`` maps logical pages to physical entries in ``k_pages`` and
    ``v_pages``. This Op reads the cache only: allocation, append, and mutation
    remain runtime responsibilities. The shell has no BUILTIN kernel yet.
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
        out_dtype: Optional[torch.dtype] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        *,
        target: Target = None,
    ) -> None:
        """Configure paged GQA semantics without owning or mutating the cache.

        Args:
            is_causal: Apply a bottom-right-aligned causal mask per request.
            window_size_left: Visible keys to the left; ``-1`` is unlimited.
            window_size_right: Visible keys to the right; ``-1`` is unlimited.
            sm_scale: Score scale, or ``None`` for ``1 / sqrt(head_dim)``.
            softcap: Positive score cap; ``None`` or zero disables it.
            pos_encoding_mode: ``"none"`` or ``"rope"``.
            rotary_dim: Even rotated width; ``None`` uses the full head dimension.
            rope_layout: ``"neox"`` or ``"interleaved"``.
            out_dtype: Output dtype, inferred from the input when omitted.
            kernel_map: Optional in-tree kernel overrides.
            target: Backend target, or ``None`` to resolve from the input device.
        """
        if window_size_left < -1:
            raise ValueError("window_size_left must be -1 (unlimited) or >= 0")
        if window_size_right < -1:
            raise ValueError("window_size_right must be -1 (unlimited) or >= 0")
        if sm_scale is not None and not math.isfinite(sm_scale):
            raise ValueError(f"sm_scale must be finite, got {sm_scale}")
        if pos_encoding_mode not in ("none", "rope"):
            raise ValueError("pos_encoding_mode must be 'none' or 'rope'")
        if rotary_dim is not None and pos_encoding_mode != "rope":
            raise ValueError("rotary_dim requires pos_encoding_mode='rope'")
        if rotary_dim is not None and (rotary_dim <= 0 or rotary_dim % 2):
            raise ValueError("rotary_dim must be a positive even integer")
        if rope_layout not in ("neox", "interleaved"):
            raise ValueError("rope_layout must be 'neox' or 'interleaved'")
        if out_dtype is not None:
            _validate_attention_dtype(out_dtype)

        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_pages_shape: tuple[int, ...],
        v_pages_shape: tuple[int, ...],
        page_table_shape: tuple[int, ...],
        cache_seqlens_shape: tuple[int, ...],
        cu_seqlens_q_shape: tuple[int, ...],
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
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> None:
        fp8 = fp8_dtype()
        if q.dtype not in (torch.float16, torch.bfloat16, fp8):
            raise ValueError("q must have float16, bfloat16, or float8_e4m3fn dtype")
        if k_pages.dtype not in (torch.float16, torch.bfloat16, fp8):
            raise ValueError("k_pages must have a supported attention dtype")
        if v_pages.dtype != k_pages.dtype:
            raise ValueError("k_pages and v_pages must have the same dtype")
        if q.dtype != fp8 and k_pages.dtype != fp8 and k_pages.dtype != q.dtype:
            raise ValueError("16-bit q and KV pages must have the same dtype")
        if q.dtype == fp8 and k_pages.dtype != fp8:
            raise ValueError("FP8 q requires FP8 KV pages")

        for name, tensor in (
            ("page_table", page_table),
            ("cache_seqlens", cache_seqlens),
            ("cu_seqlens_q", cu_seqlens_q),
        ):
            if tensor.dtype != torch.int32:
                raise ValueError(f"{name} must have int32 dtype")
        for name, tensor in (("q_scale", q_scale), ("k_scale", k_scale), ("v_scale", v_scale)):
            if tensor is not None and tensor.dtype != torch.float32:
                raise ValueError(f"{name} must have float32 dtype")
        for name, tensor in (("rope_cos", rope_cos), ("rope_sin", rope_sin)):
            if tensor is not None and tensor.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(f"{name} must have float16 or bfloat16 dtype")

    def eval_roofline(self) -> tuple[int, int]:
        raise NotImplementedError("Paged GQA has no in-tree implementation yet")

    def _validate_forward_inputs(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
    ) -> None:
        if q.ndim != 3:
            raise ValueError("q must be a rank-3 THD tensor")
        if k_pages.ndim != 4 or v_pages.shape != k_pages.shape:
            raise ValueError("k_pages and v_pages must share rank-4 paged layout")
        if page_table.ndim != 2:
            raise ValueError("page_table must be rank 2")
        if cache_seqlens.ndim != 1:
            raise ValueError("cache_seqlens must be rank 1")
        batch = cache_seqlens.shape[0]
        if page_table.shape[0] != batch:
            raise ValueError("page_table and cache_seqlens must have the same batch size")
        if cu_seqlens_q.shape != (batch + 1,):
            raise ValueError(f"cu_seqlens_q must have shape {(batch + 1,)}")

        _, heads, dim = q.shape
        _, page_size, heads_kv, dim_kv = k_pages.shape
        if page_size <= 0:
            raise ValueError("KV page size must be positive")
        if dim_kv != dim:
            raise ValueError("q and KV pages must have the same head dimension")
        _validate_gqa_dims(heads, heads_kv, dim)

        self._validate_dtypes(
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )

        fp8 = fp8_dtype()
        q_is_fp8 = q.dtype == fp8
        output_dtype = self.out_dtype or q.dtype
        if q_is_fp8 and self.out_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FP8 q requires a 16-bit output dtype")
        if not q_is_fp8 and output_dtype != q.dtype:
            raise ValueError("16-bit output dtype must match q")

        if q_is_fp8 != (q_scale is not None):
            raise ValueError("q_scale is required exactly when q is FP8")
        if (k_scale is None) != (v_scale is None):
            raise ValueError("k_scale and v_scale must be supplied together")
        kv_is_fp8 = k_pages.dtype == fp8
        if kv_is_fp8 != (k_scale is not None):
            raise ValueError("k_scale and v_scale are required exactly when KV pages are FP8")

        tensors = (
            ("k_pages", k_pages),
            ("v_pages", v_pages),
            ("page_table", page_table),
            ("cache_seqlens", cache_seqlens),
            ("cu_seqlens_q", cu_seqlens_q),
        )
        for name, tensor in tensors:
            if tensor.device != q.device:
                raise ValueError(f"{name} must be on the same device as q")
        if q_scale is not None and (
            q_scale.device != q.device or tuple(q_scale.shape) != (batch, heads_kv)
        ):
            raise ValueError(f"q_scale must have shape {(batch, heads_kv)} on q.device")
        for name, scale in (("k_scale", k_scale), ("v_scale", v_scale)):
            if scale is None:
                continue
            valid_shape = tuple(scale.shape) in ((1,), (batch, heads_kv))
            if scale.device != q.device or not valid_shape:
                raise ValueError(f"{name} must have shape (1,) or {(batch, heads_kv)} on q.device")

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
            if table.dtype != output_dtype:
                raise ValueError(f"{name} must have dtype {output_dtype}")
            if table.ndim != 2 or table.shape[0] < 1 or table.shape[1] != expected_columns:
                raise ValueError(f"{name} must have shape [max_position, {expected_columns}]")
        if rope_cos.shape != rope_sin.shape:
            raise ValueError("rope_cos and rope_sin must have the same shape")

    @staticmethod
    def _canonicalize_inputs(
        *inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        return tuple(tensor.contiguous() if tensor is not None else None for tensor in inputs)

    def forward(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run read-only paged GQA over packed Q and rank-4 KV pages."""
        self._validate_forward_inputs(
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        inputs = self._canonicalize_inputs(
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        kernel = self.kernel_for("gqa_paged", inputs)
        return kernel(*inputs)


class GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(Op):
    """Packed GQA prefill with paged KV cache append. Layout: THD.

    The current chunk is packed by request. ``cache_seqlens`` stores each
    request's logical KV length before append. ``block_table`` maps logical
    page ids to physical pages in ``k_pages`` / ``v_pages``.

    The in-tree kernels refuse a ``page_size`` that is not a power of two, fused RoPE
    over an FP8 cache, FP8 cache scales that are not finite and positive, and a
    fused-RoPE call whose cached plus new tokens exceed ``max_position``.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_prefill_paged_with_kv_cache_fwd_kernel": GQAPrefillPagedWithKVCacheFwdKernel,
        "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel": GQAPrefillPagedWithFP8KVCacheFwdKernel,
        "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel": GQAPrefillPagedWithKVCacheRopeFwdKernel,
    }

    def eval_roofline_read_bytes(self) -> "int | None":
        """Not derivable here: the call writes part of the pool, not all of it.

        ``k_pages`` and ``v_pages`` are mutated, and the base class takes a
        mutated input's whole extent off ``bytes`` as the write. This call
        appends the new tokens into pages the block table names and leaves the
        rest untouched, so that subtraction would understate the read half.
        """
        return None

    def roofline_inputs(self) -> "dict[str, int]":
        """The cached tokens this call reads, which its cache traffic follows."""
        from tileops.perf.formulas import gqa_prefill_paged_cached_tokens

        return {"cached_tokens": gqa_prefill_paged_cached_tokens(self.last_call)}

    def __init__(
        self,
        page_size: int,
        max_seqlen_q: int,
        is_causal: bool = True,
        cache_dtype: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        fuse_rope: bool = False,
        rope_base: float = 10000.0,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            max_seqlen_q: Manifest ``params.max_seqlen_q``, the launch bound the kernel
                is built for; a call whose longest request exceeds it is refused.
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            cache_dtype: Manifest ``params.cache_dtype``, ``dtype | None``, default ``None``.
            sm_scale: Manifest ``params.sm_scale``, ``float | None``, default ``None``,
                which resolves to ``1 / sqrt(D)`` from each call's head dimension.
            softcap: Manifest ``params.softcap``, ``float | None``, default ``None``.
            fuse_rope: Manifest ``params.fuse_rope``, ``bool``, default ``False``.
            rope_base: Manifest ``params.rope_base``, ``float``, default ``10000.0``.
            max_position: Manifest ``params.max_position``, ``int | None``, default ``None``.
            rotary_dim: Manifest ``params.rotary_dim``, ``int | None``, default ``None``,
                which rotates each call's full head dimension.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.max_seqlen_q = max_seqlen_q
        self.page_size = page_size
        self.is_causal = is_causal
        # None means the cache holds whatever element type forward is given.
        self.cache_dtype = cache_dtype
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.fuse_rope = fuse_rope
        self.rope_base = rope_base
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._rope_cos_cache: Dict[
            tuple[torch.device, torch.dtype, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def _resolved_cache_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Cache element type for an attention element type of *dtype*."""
        return dtype if self.cache_dtype is None else self.cache_dtype

    def _resolved_rotary_dim(self, dim: int) -> Optional[int]:
        """Rotated width for a head dimension of *dim*, or ``None`` without fused RoPE."""
        return _rope_rotary_dim(dim, self.rotary_dim) if self.fuse_rope else None

    def attention_call(
        self, q: torch.Tensor, k_new: torch.Tensor, block_table: torch.Tensor
    ) -> AttentionCall:
        """State what one paged prefill call is, for selection to filter against."""
        _, heads, dim = q.shape
        heads_kv = k_new.shape[1]
        batch, max_pages_per_req = block_table.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_pages_per_req=max_pages_per_req,
            page_size=self.page_size,
            is_causal=self.is_causal,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
            cache_dtype=self._resolved_cache_dtype(q.dtype),
            fuse_rope=self.fuse_rope,
            max_position=self.max_position,
            rotary_dim=self._resolved_rotary_dim(dim),
            tune=self.tune,
            device=q.device,
        )

    def _rope_tables(self, q: torch.Tensor):
        """Rotary tables for this call, or ``(None, None)`` when the op fuses no RoPE."""
        if not self.fuse_rope:
            return None, None
        return self._get_rope_cos_sin(q.device, q.dtype, self._resolved_rotary_dim(q.shape[2]))

    def _check_call_values(
        self,
        k_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> None:
        """Refuse the calls the in-tree kernels cannot serve.

        The kernels index pages by shift, so ``page_size`` is a power of two; the RoPE
        kernel appends a 16-bit cache only. An FP8 cache dequantizes by the scales, which
        must be finite and positive; fused RoPE indexes its table by position, which must
        stay below ``max_position``.
        """
        if self.page_size & (self.page_size - 1) != 0:
            raise ValueError("page_size must be a power of two")
        if self.fuse_rope and k_pages.dtype == fp8_dtype():
            raise ValueError("fuse_rope is not supported with FP8 paged KV cache yet")
        if k_pages.dtype == fp8_dtype():
            for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
                if not torch.all(torch.isfinite(tensor) & (tensor > 0)).item():
                    raise ValueError(f"{name} must contain finite positive values")
        if self.fuse_rope:
            q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_total_len = int((cache_seqlens + q_lens).max().item())
            if max_total_len > self.max_position:
                raise ValueError(
                    "cache_seqlens + q_len exceeds RoPE max_position: "
                    f"max total length {max_total_len}, max_position {self.max_position}"
                )

    def _get_rope_cos_sin(
        self,
        device: torch.device,
        dtype: torch.dtype,
        rotary_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_position is None:
            raise ValueError("max_position is required when fuse_rope=True")
        key = (device, dtype, rotary_dim)
        cached = self._rope_cos_cache.get(key)
        if cached is None:
            cached = base_freqs(
                rotary_dim,
                self.max_position,
                base=self.rope_base,
                dtype=dtype,
                device=device,
            )
            self._rope_cos_cache[key] = cached
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
    ) -> torch.Tensor:
        """Attend the packed chunk to each request's cache, appending its keys and values.

        Args:
            q: Queries of the new tokens, packed by request [total_q, heads, dim].
            k_new: Keys of the new tokens [total_q, heads_kv, dim].
            v_new: Values of the new tokens [total_q, heads_kv, dim].
            k_pages: Key page pool [physical_tokens, heads_kv, dim], written in place.
            v_pages: Value page pool [physical_tokens, heads_kv, dim], written in place.
            k_scale: Key dequantization scale of an FP8 pool [1].
            v_scale: Value dequantization scale of an FP8 pool [1].
            cu_seqlens_q: Request boundaries in the packed chunk [batch + 1].
            cache_seqlens: Each request's cache length before the append [batch].
            block_table: Physical page of each request's logical page [batch, pages].

        Returns:
            The attention output [total_q, heads, dim].

        Raises:
            ValueError: An FP8 pool's scales are not finite and positive, or a fused-RoPE
                call reaches past ``max_position``.
        """
        return self._call_boundary(
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
        )

    def _eager_forward(
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
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._check_call_values(k_pages, k_scale, v_scale, cu_seqlens_q, cache_seqlens)
        q, k_new, v_new, k_scale, v_scale, cu_seqlens_q, cache_seqlens, block_table = (
            t.contiguous()
            for t in (q, k_new, v_new, k_scale, v_scale, cu_seqlens_q, cache_seqlens, block_table)
        )
        call = self.attention_call(q, k_new, block_table)
        cos_table, sin_table = self._rope_tables(q)
        inputs = (
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
        )
        kernel = self.kernel_for("gqa_prefill_paged", inputs, call)
        return kernel(*inputs, self.max_seqlen_q, cos_table, sin_table)

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
        return tensor_core_roof(self.last_call.tensors["q"][1])


class GroupedQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_bwd_preprocess_kernel": FlashAttnBwdPreprocessKernel,
        "gqa_bwd_kernel": GQABwdWgmmaPipelinedKernel,
    }

    def __init__(
        self,
        is_causal: bool = True,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            is_causal: Manifest ``params.is_causal``, ``bool``, default ``True``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.is_causal = is_causal

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _get_kernels(
        self, inputs: "tuple[torch.Tensor | None, ...]", call: tuple
    ) -> tuple[Kernel, Kernel]:
        """Return (preprocess, backward) kernels for *call*, building once each."""
        return self.kernel_for("gqa_bwd", inputs, call)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """Both passes run on every call, so they are built together as one entry, per
        ``(batch, heads, heads_kv, seq_len, dim, dtype)``."""
        batch, heads, heads_kv, seq_len, dim, dtype = call

        def build() -> tuple[Kernel, Kernel]:
            return (
                self.kernel_map["gqa_bwd_preprocess_kernel"](
                    batch,
                    heads,
                    seq_len,
                    dim,
                    dtype,
                    tune=self.tune,
                ),
                self.kernel_map["gqa_bwd_kernel"](
                    batch,
                    heads,
                    heads_kv,
                    seq_len,
                    dim,
                    self.is_causal,
                    dtype,
                    tune=self.tune,
                ),
            )

        return call, build

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
        return self._call_boundary(q, k, v, o, do, lse)

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        do: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        do = do.contiguous()
        batch, seq_len, heads, dim = q.shape
        prep_kernel, kernel = self._get_kernels(
            (q, k, v, o, do, lse), (batch, heads, k.shape[2], seq_len, dim, q.dtype)
        )
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
        return tensor_core_roof(self.last_call.tensors["q"][1])


class GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged GQA decode with dynamic KV cache. Layout: ``Q`` $[batch \\times heads \\times dim]$ (BHD);
    K, V physical cache [seqlen_kv, heads_kv, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].

    The in-tree kernels refuse a ``page_size`` that no supported key tile width divides.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_decode_paged_kernel": GQADecodePagedKernel,
        "gqa_decode_paged_bs1_kernel": GQADecodePagedBs1Kernel,
    }

    def __init__(
        self,
        page_size: int,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            page_size: Manifest ``params.page_size``, ``int``.
            sm_scale: Manifest ``params.sm_scale``, ``float | None``, default ``None``,
                which resolves to ``1 / sqrt(D)`` from each call's head dimension.
            softcap: Manifest ``params.softcap``, ``float | None``, default ``None``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.target = target
        self.page_size = page_size
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)

        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def attention_call(self, q: torch.Tensor, k: torch.Tensor) -> AttentionCall:
        """State what one paged decode call is, for selection to filter against."""
        batch, heads, dim = q.shape
        seqlen_kv, heads_kv, _ = k.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seqlen_kv=seqlen_kv,
            dim=dim,
            page_size=self.page_size,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
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
            k: Input tensor, dtype ``same_as(q)``.
            v: Input tensor, dtype ``same_as(q)``.
            real_seqlen_kv: Input tensor, dtype ``int32``.
            block_table: Input tensor, dtype ``int32``.

        Returns:
            ``o``, as the manifest declares. Shape rules: ``o.shape == (B, H, D)``.
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
        kernel = self.kernel_for("gqa_decode_paged", inputs, self.attention_call(q, k))
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
