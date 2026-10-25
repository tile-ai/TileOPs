from typing import ClassVar, Mapping, Optional

import torch

from tileops._kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADecodeWSKernel,
    GQADenseFP8DecodeKernel,
    GQADenseFP8Kernel,
    GQADensePersistentWSKernel,
    GQADensePrefillFwdKernel,
    GQADenseSlidingWindowBandKernel,
    GQADenseSlidingWindowKernel,
    GQADenseWSKernel,
)
from tileops._kernels.attention.call_spec import (
    AttentionCall,
    GQADenseFwdInterface,
)
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops.attention.gqa.parameters import rope_rotary_dim, score_softcap
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["GQADenseFwdOp", "GQADenseWithLSEFwdOp"]


class _GQADenseFwdOpBase(Op):
    """The kernels and call spec :class:`GQADenseFwdOp` and :class:`GQADenseWithLSEFwdOp`
    share.

    A subclass assigns ``is_causal``, ``window_size_left``, ``window_size_right``,
    ``sm_scale`` and ``softcap``; the class defaults below stand for the RoPE and FP8
    parameters an op without them leaves off.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_dense_prefill": GQADensePrefillFwdKernel,
        "gqa_dense": GQADenseWSKernel,
        "gqa_dense_persistent": GQADensePersistentWSKernel,
        "gqa_dense_decode": GQADecodeKernel,
        "gqa_dense_decode_bs1": GQADecodeBs1Kernel,
        "gqa_dense_fp8": GQADenseFP8Kernel,
        "gqa_dense_fp8_decode": GQADenseFP8DecodeKernel,
        "gqa_dense_decode_long_context": GQADecodeLongContextKernel,
        "gqa_dense_decode_ws": GQADecodeWSKernel,
        "gqa_dense_sliding_window": GQADenseSlidingWindowKernel,
        "gqa_dense_sliding_window_band": GQADenseSlidingWindowBandKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gqa_dense": GQADenseFwdInterface}

    # Whether the kernel also returns each query row's log-sum-exp.
    _returns_lse: ClassVar[bool] = False
    pos_encoding_mode: str = "none"
    rotary_dim: Optional[int] = None
    rope_layout: str = "neox"
    out_dtype: Optional[torch.dtype] = None

    def roof_key(self) -> str:
        """Dense attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def _dense_call(self, inputs: tuple[Optional[torch.Tensor], ...]) -> AttentionCall:
        """State what one contiguous call is, for selection to filter against."""
        q, k, _v, _q_scale, _k_scale, _v_scale, rope_cos, _rope_sin, sinks = inputs
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
            has_sinks=sinks is not None,
            fuse_rope=rope_on,
            max_position=rope_cos.shape[0] if rope_cos is not None else 1,
            rotary_dim=rope_rotary_dim(dim, self.rotary_dim) if rope_on else 0,
            rope_layout=self.rope_layout,
            return_lse=self._returns_lse,
            device=q.device,
        )

    def _run_dense(self, *tensors: Optional[torch.Tensor]):
        """Run the kernel serving the nine tensor slots, each made contiguous."""
        inputs = tuple(tensor.contiguous() if tensor is not None else None for tensor in tensors)
        return self.kernel_for("gqa_dense", lambda: self._dense_call(inputs))(*inputs)


class GQADenseFwdOp(_GQADenseFwdOpBase):
    r"""Grouped-Query Attention (GQA) over dense $Q$/$K$/$V$ tensors, layout BSHD.

    Every batch entry has one query length $S_q$ and one KV length $S_{kv}$. The $H$ query
    heads form $H_{kv}$ groups of $g = H / H_{kv}$; head $h$ reads KV head
    $r(h) = \lfloor h / g \rfloor$. Query row $i$ sits at KV position
    $p_i = i + S_{kv} - S_q$. With $\alpha$ = ``sm_scale`` (default $1/\sqrt D$),
    $c$ = ``softcap`` and $\sigma_h = \exp(\mathrm{sinks}_h)$, or $0$ without ``sinks``:

    $$
    \begin{aligned}
    Z_{h,i,j} &= \alpha\,\langle Q'_{i,h}, K'_{j,r(h)} \rangle, \qquad
    L_{h,i,j} = \begin{cases}
        c\tanh(Z_{h,i,j}/c) \text{ if } c > 0, \text{ else } Z_{h,i,j}, & j \in \mathcal V_i, \\
        -\infty, & j \notin \mathcal V_i,
        \end{cases} \\
    O_{i,h} &= \sum_j \frac{\exp L_{h,i,j}}{\sigma_h + \sum_k \exp L_{h,i,k}}\,V'_{j,r(h)}.
    \end{aligned}
    $$

    Key $j$ is in the visible set $\mathcal V_i$ when ``is_causal`` is off or $j \le p_i$,
    and it lies within ``window_size_left`` keys left and ``window_size_right`` keys right
    of $p_i$, ``-1`` meaning unlimited. $Q'$ and $K'$ are $Q$ and $K$ times ``q_scale`` and
    ``k_scale`` for FP8 input, rotated under ``pos_encoding_mode="rope"`` at positions $p_i$
    and $j$; $V'$ is $V$ times ``v_scale``. A sink is neither scaled nor capped, and the
    in-tree kernels serve ``sinks`` on SM90 only. Accumulation is FP32; the output takes the
    input dtype, or ``out_dtype`` for FP8 input.
    """

    def __init__(
        self,
        is_causal: bool = False,
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
    ) -> None:
        r"""Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Apply the causal mask, bottom-right aligned. Default
                ``False``, as in ``flash_attn_func``.
            window_size_left: Keys admitted left of the query row's KV
                position $p_i$; ``-1`` means unlimited.
            window_size_right: Keys admitted right of $p_i$; ``-1`` means
                unlimited.
            sm_scale: Score scale $\alpha$. ``None`` means $1 / \sqrt{D}$, $D$
                the head dimension of ``q`` and ``k`` in the current call.
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

        Raises:
            ValueError: ``softcap`` is negative.
        """
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        # A ``None`` scale depends on D, so it stays None here and resolves
        # when an implementation is requested for the current call.
        self.sm_scale = sm_scale
        self.softcap = score_softcap(softcap)
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.out_dtype = out_dtype
        super().__init__(target=target)

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
        sinks: Optional[torch.Tensor] = None,
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
            sinks: Optional ``float32`` logits, shape ``[H]``, shared across
                batches and query positions. Each contributes no value.
                Supported by in-tree kernels on SM90 only, for prefill and decode.

        Returns:
            Attention output, $[B \times S_q \times H \times D]$, laid out
            like ``q`` and contiguous. Its dtype is ``out_dtype`` for FP8 input,
            and the input dtype otherwise.

        Raises:
            ValueError: Shapes, dtypes, devices, or optional-input
                combinations violate the contract above, or no in-tree kernel
                serves the call; the message names the limit each kernel refused.
        """
        return self._run_dense(q, k, v, q_scale, k_scale, v_scale, rope_cos, rope_sin, sinks)


class GQADenseWithLSEFwdOp(_GQADenseFwdOpBase):
    r"""Dense Grouped-Query Attention (GQA) forward that also returns each query row's
    log-sum-exp, the pair :class:`GQADenseBwdOp` takes.

    It computes the output of :class:`GQADenseFwdOp` over 16-bit $Q$/$K$/$V$ with the
    same causal mask, window, score scale and softcap, and adds

    $$
    \mathrm{lse}_{h,i} = \log \sum_{j \in \mathcal V_i} \exp(L_{h,i,j}),
    $$

    the natural-log log-sum-exp of the scaled, capped and masked logits. Its reference is
    ``flash_attn_interface._flash_attn_forward``, the forward of FlashAttention-3's
    training path; ``softmax_scale`` is ``sm_scale`` here and ``causal`` is ``is_causal``.

    The in-tree kernels serve it on SM90 for prefill ($S_q > 1$) without a window; any
    other call is refused when a kernel is selected.
    """

    _returns_lse: ClassVar[bool] = True

    def __init__(
        self,
        is_causal: bool = False,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        *,
        target: Target = None,
    ) -> None:
        r"""Configure the op. Tensor shapes and input dtype come from each call.

        Args:
            is_causal: Apply the causal mask, bottom-right aligned.
            window_size_left: Keys admitted left of the query row's KV position; ``-1``
                means unlimited.
            window_size_right: Keys admitted right of it; ``-1`` means unlimited.
            sm_scale: Score scale. ``None`` means $1 / \sqrt{D}$, $D$ the head dimension
                of ``q`` and ``k`` in the current call.
            softcap: Positive cap $c$ replacing each raw score $z$ with $c \tanh(z / c)$.
                ``None`` or ``0`` disables capping.
            target: Backend target to serve this op, or ``None`` to decide from the input
                device.

        Raises:
            ValueError: ``softcap`` is negative.
        """
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.sm_scale = sm_scale
        self.softcap = score_softcap(softcap)
        super().__init__(target=target)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Run dense GQA attention and return its output with each row's log-sum-exp.

        Args:
            q: Queries, $[B \times S_q \times H \times D]$; ``float16`` or ``bfloat16``.
            k: Keys, $[B \times S_{kv} \times H_{kv} \times D]$, same dtype as ``q``.
            v: Values, $[B \times S_{kv} \times H_{kv} \times D]$, same dtype as ``q``.

        Returns:
            ``(o, lse)``. ``o`` is the attention output, $[B \times S_q \times H \times D]$
            in the input dtype. ``lse`` is ``float32``, $[B \times H \times S_q]$, ``-inf``
            for a row that sees no key.

        Raises:
            ValueError: Shapes, dtypes or devices violate the contract above, or no
                in-tree kernel serves the call; the message names the limit each kernel
                refused.
        """
        return self._run_dense(q, k, v, *([None] * 6))
