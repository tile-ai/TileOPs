from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADenseFP8DecodeKernel,
    GQADenseFP8Kernel,
    GQADenseSlidingWindowKernel,
    GQADenseWSKernel,
)
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQADenseFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.attention.gqa.parameters import _rope_rotary_dim, _score_softcap
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GQADenseFwdOp"]


class GQADenseFwdOp(Op):
    r"""Grouped-Query Attention (GQA) over dense $Q$/$K$/$V$ tensors.

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
        "gqa_dense": GQADenseWSKernel,
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
        return self.kernel_for("gqa_dense", self.dense_call(inputs))(*inputs)
