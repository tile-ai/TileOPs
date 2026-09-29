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
    MHABwdWsKernel,
)
from tileops.kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.ops.rope import base_freqs
from tileops.perf.profile import tensor_core_roof

__all__ = [
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDenseFwdOp",
    "GroupedQueryAttentionPagedFwdOp",
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp",
    "GroupedQueryAttentionVarlenFwdOp",
]


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
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gqa_dense": GQADenseFwdInterface}

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
            ValueError: ``softcap`` is negative.
        """
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

    def dense_call(self, inputs: tuple[Optional[torch.Tensor], ...]) -> AttentionCall:
        """State what one contiguous call is, for selection to filter against."""
        q, k, _v, _q_scale, _k_scale, _v_scale, rope_cos, _rope_sin = inputs
        assert q is not None and k is not None
        batch, seq_len_q, heads, dim = q.shape
        _, seq_len_kv, heads_kv, _ = k.shape
        rope_on = self.pos_encoding_mode == "rope"
        is_fp8 = q.dtype == torch.float8_e4m3fn
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
                combinations violate the contract above, or no in-tree kernel
                serves the call; the message names the limit each kernel refused.
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
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
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
            target: Backend target, or ``None`` to resolve from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.validate_inputs = validate_inputs
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_varlen": GQAPrefillVarlenFwdKernel,
            "gqa_varlen_ws": GQAPrefillVarlenWSFwdKernel,
            "gqa_varlen_sliding_window": GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
        }

    def compute_roof(self) -> str:
        """Varlen attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

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
            is_fp8=q.dtype == torch.float8_e4m3fn,
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

    def _check_offsets(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
        """Check on the CPU that the offsets span the packed tensors and never decrease."""
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
        """Resolve the implementation and launch it."""
        if self.validate_inputs:
            self._check_offsets(q, k, cu_seqlens_q, cu_seqlens_kv)
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
        return self._get_kernel(inputs)(*inputs)


class GroupedQueryAttentionPagedFwdOp(Op):
    """Grouped-query attention over a caller-owned paged KV cache.

    Packed Q and its cumulative sequence lengths cover both prefill and decode.
    ``page_table`` maps logical pages to physical entries in ``k_pages`` and
    ``v_pages``. This Op reads the cache only: allocation, append, and mutation
    remain runtime responsibilities. The in-tree kernels serve a call in which
    every request carries the same number of query tokens, Q and KV share a
    float16 or bfloat16 dtype, and no window, RoPE or FP8 is requested; they
    refuse any other call.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_decode_paged_kernel": GQADecodePagedKernel,
        "gqa_decode_paged_bs1_kernel": GQADecodePagedBs1Kernel,
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The cached tokens this call's lengths name and the distinct pool rows it reads,
        which its flops and cache reads follow."""
        from tileops.perf.formulas import gqa_paged_cache_rows

        call = self.last_call
        return {
            "cached_tokens": sum(call.values("cache_seqlens")),
            "cache_rows": gqa_paged_cache_rows(call),
        }

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
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure paged GQA semantics without owning or mutating the cache.

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
            target: Backend target, or ``None`` to resolve from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
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
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """Paged attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def paged_call(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        page_table: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> AttentionCall:
        """State what one paged call is, for selection to filter against.

        The query lengths are read from every step of ``cu_seqlens_q``: a packed
        total equal to the batch does not by itself mean one token per request.
        """
        _, heads, dim = q.shape
        num_pages, page_size, heads_kv, _ = k_pages.shape
        batch, max_pages_per_req = page_table.shape
        q_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            max_seqlen_q=max(q_lens, default=0),
            seqlen_kv=num_pages * page_size,
            page_size=page_size,
            max_pages_per_req=max_pages_per_req,
            is_causal=self.is_causal,
            sm_scale=_attention_scale(dim, self.sm_scale),
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=torch.float8_e4m3fn in (q.dtype, k_pages.dtype),
            is_uniform=len(set(q_lens)) <= 1,
            cache_dtype=k_pages.dtype,
            fuse_rope=self.pos_encoding_mode == "rope",
            tune=self.tune,
            device=q.device,
        )

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
        """Run read-only paged GQA over packed Q and rank-4 KV pages.

        Args:
            q: Packed queries [total_q, heads, dim].
            k_pages: Key pages [num_pages, page_size, heads_kv, dim].
            v_pages: Value pages [num_pages, page_size, heads_kv, dim].
            page_table: Physical page of each request's logical page [batch, pages].
            cache_seqlens: Each request's cached length, its query tokens included [batch].
            cu_seqlens_q: Request boundaries in the packed queries [batch + 1].
            q_scale: Dequantization scale of an FP8 query [batch, heads_kv].
            k_scale: Dequantization scale of an FP8 key cache.
            v_scale: Dequantization scale of an FP8 value cache.
            rope_cos: Rotary cosine table when ``pos_encoding_mode='rope'``.
            rope_sin: Rotary sine table when ``pos_encoding_mode='rope'``.

        Returns:
            The attention output [total_q, heads, dim].
        """
        return self._call_boundary(
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

    def _eager_forward(
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
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder. The kernels read
        the pool as ``[num_pages * page_size, heads_kv, dim]``, a view of the pages.
        """
        q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q = (
            t.contiguous() for t in (q, k_pages, v_pages, page_table, cache_seqlens, cu_seqlens_q)
        )
        inputs = (
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
        call = self.paged_call(q, k_pages, page_table, cu_seqlens_q)
        kernel = self.kernel_for("gqa_paged", inputs, call)
        return kernel(q, k_pages.flatten(0, 1), v_pages.flatten(0, 1), cache_seqlens, page_table)


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
        """The distinct cache rows this call reads, which its cache traffic follows."""
        from tileops.perf.formulas import gqa_prefill_paged_cache_rows

        return {"cache_rows": gqa_prefill_paged_cache_rows(self.last_call)}

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
        """Refuse tensor contents the call is undefined for.

        An FP8 cache dequantizes by the scales, which must be finite and positive; fused
        RoPE indexes its table by position, which must stay below ``max_position``.
        """
        if k_pages.dtype == torch.float8_e4m3fn:
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
            ValueError: An FP8 pool's scales are not finite and positive, a fused-RoPE
                call reaches past ``max_position``, or no in-tree kernel serves the call.
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
        "gqa_bwd_ws_kernel": MHABwdWsKernel,
    }
    _BACKWARD_KEYS = ("gqa_bwd_kernel", "gqa_bwd_ws_kernel")

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

    def _attention_call(self, q: torch.Tensor, k: torch.Tensor) -> AttentionCall:
        """State what one backward call is, for selection to filter against."""
        batch, seq_len, heads, dim = q.shape
        return AttentionCall(
            dtype=q.dtype,
            batch=batch,
            heads=heads,
            heads_kv=k.shape[2],
            dim=dim,
            max_seqlen_q=seq_len,
            seqlen_kv=seq_len,
            is_causal=self.is_causal,
            tune=self.tune,
            device=q.device,
        )

    def entry_for(self, role: str, call: AttentionCall) -> Entry:
        """The preprocess pass has one implementation, built per ``(batch, heads,
        seq_len, dim, dtype)``; the backward pass is chosen among its candidates."""
        if role == "gqa_bwd_preprocess":
            args = (call.batch, call.heads, call.max_seqlen_q, call.dim, call.dtype)
            cls = self.kernel_map["gqa_bwd_preprocess_kernel"]
            return (cls, args), lambda: cls(*args, tune=self.tune)
        cls = self.select_kernel(call, self._BACKWARD_KEYS)
        identity, build = cls.entry_for(call)
        return (cls, identity), build

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
            k: Input tensor, same dtype as ``q``.
            v: Input tensor, same dtype as ``q``.
            o: Input tensor, same dtype as ``q``.
            do: Input tensor, same dtype as ``q``.
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
        inputs = (q, k, v, o, do, lse)
        call = self._attention_call(q, k)
        prep_kernel = self.kernel_for("gqa_bwd_preprocess", inputs, call)
        kernel = self.kernel_for("gqa_bwd", inputs, call)
        delta, dq_accum = prep_kernel(o, do)
        return kernel(q, k, v, do, lse, delta, dq_accum)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
