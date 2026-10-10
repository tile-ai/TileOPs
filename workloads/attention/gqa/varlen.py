import math

import torch

from workloads.attention.gqa.call_metadata import _dtype, _segments
from workloads.attention.gqa.rope import apply_dense_rope, apply_packed_rope
from workloads.device import run_device
from workloads.sequence_metadata import make_cu_seqlens
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "GQAPrefillVarlenFwdWorkload",
    "GQASlidingWindowVarlenFwdWorkload",
    "GQAVarlenCall",
    "GQAVarlenFwdWorkload",
    "GQAVarlenScaledCall",
    "GQAVarlenScaledWorkload",
]


class GQAPrefillVarlenFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        q_lens: list[int],
        kv_lens: list[int],
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.q_lens = q_lens
        self.kv_lens = kv_lens
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def total_kv(self) -> int:
        return sum(self.kv_lens)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.q_lens)

    @property
    def max_seqlen_kv(self) -> int:
        return max(self.kv_lens)

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.total_q, self.heads, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        k = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        v = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=run_device(),
        )
        cu_seqlens_kv = torch.tensor(
            [0] + torch.tensor(self.kv_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=run_device(),
        )
        return q, k, v, cu_seqlens_q, cu_seqlens_kv


class GQAVarlenFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        seqlens_q: list[int],
        seqlens_k: list[int],
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
        sm_scale: float | None = None,
        softcap: float | None = None,
        pos_encoding_mode: str = "none",
        rotary_dim: int | None = None,
        rope_layout: str = "neox",
    ) -> None:
        self.batch = batch
        self.seqlens_q = seqlens_q
        self.seqlens_k = seqlens_k
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.wl = wl
        self.wr = wr
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = dim if rotary_dim is None else rotary_dim
        self.rope_layout = rope_layout

    @property
    def max_seqlen_q(self) -> int:
        return max(self.seqlens_q)

    @property
    def max_seqlen_kv(self) -> int:
        return max(self.seqlens_k)

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        """The five packed tensors, plus the RoPE tables when the call carries them."""
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        # Unit-scale Q/K keep softmax away from a near-uniform distribution;
        # otherwise ignoring Q can fit inside the storage-dtype error bound.
        q = torch.randn(total_q, self.heads, self.dim, dtype=self.dtype, device=run_device())
        k = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device=run_device())
        v = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device=run_device())

        cu_seqlens_q = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_q), 0).tolist()),
            dtype=torch.int32,
            device=run_device(),
        )
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_k), 0).tolist()),
            dtype=torch.int32,
            device=run_device(),
        )
        if self.pos_encoding_mode != "rope":
            return q, k, v, cu_seqlens_q, cu_seqlens_k
        # Angles span a whole turn. A narrow draw puts every cosine near one and every sine
        # near zero, which makes the rotation near-identity: a kernel that skips it, pairs
        # the wrong channels, or rotates the channels a partial width should leave alone
        # then lands inside the tolerance and the row proves nothing.
        angles = torch.rand(
            max(max(self.seqlens_k), 1), self.rotary_dim // 2, device=run_device()
        ) * (2 * math.pi)
        cos, sin = angles.cos().to(self.dtype), angles.sin().to(self.dtype)
        return q, k, v, cu_seqlens_q, cu_seqlens_k, None, None, None, cos, sin

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Canonical materialized reference for regular and windowed Varlen GQA.

        Under RoPE, query token ``i`` of a request sits at position ``kv_len - q_len + i``
        and key token ``j`` at position ``j``; the rotation runs before the scores form.
        """
        groups = self.heads // self.heads_kv
        scale = self.dim**-0.5 if self.sm_scale is None else self.sm_scale
        outputs = []
        for request in range(self.batch):
            q_start = int(cu_seqlens_q[request].item())
            q_end = int(cu_seqlens_q[request + 1].item())
            kv_start = int(cu_seqlens_kv[request].item())
            kv_end = int(cu_seqlens_kv[request + 1].item())
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            q_b, k_b = q[q_start:q_end], k[kv_start:kv_end]
            if rope_cos is not None:
                rope = dict(rotary_dim=self.rotary_dim, layout=self.rope_layout)
                positions = torch.arange(kv_len, device=q.device)
                q_b = apply_dense_rope(
                    q_b[None], positions[kv_len - q_len :], rope_cos, rope_sin, **rope
                )[0]
                k_b = apply_dense_rope(k_b[None], positions, rope_cos, rope_sin, **rope)[0]
            q_i = q_b.transpose(0, 1).float()
            k_i = k_b.repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            v_i = v[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            scores = torch.matmul(q_i, k_i.transpose(-2, -1)) * scale
            if self.softcap is not None and self.softcap > 0:
                scores = self.softcap * torch.tanh(scores / self.softcap)
            offset = kv_len - q_len
            q_pos = torch.arange(q_len, device=q.device)[:, None] + offset
            kv_pos = torch.arange(kv_len, device=q.device)[None, :]
            visible = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device)
            if self.is_causal:
                visible &= kv_pos <= q_pos
            if self.wl >= 0:
                visible &= kv_pos >= q_pos - self.wl
            if self.wr >= 0:
                visible &= kv_pos <= q_pos + self.wr
            scores = scores.masked_fill(~visible.view(1, q_len, kv_len), float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            probs = torch.where(
                visible.any(dim=-1).view(1, q_len, 1), probs, torch.zeros_like(probs)
            )
            outputs.append(torch.matmul(probs, v_i).transpose(0, 1).to(q.dtype).contiguous())
        return torch.cat(outputs, dim=0)

    def verification(self, *inputs):
        from workloads.numerics import Exact, zeroed_input

        if inputs[0].dtype == torch.float8_e4m3fn:
            # FP8 softmax and optional fused QK rotation each round once.
            atol, rtol = (0.12, 0.04) if getattr(self, "rotary_dim", 0) else (0.08, 0.02)
        else:
            atol = rtol = 1e-3 if inputs[0].dtype == torch.float16 else 1e-2
        # Query removal is observable only when attention can select among keys.
        # Empty KV and one-key/window-only-self cases are independent of Q by definition.
        control = (
            self.sm_scale != 0
            and self.wl != 0
            and any(
                q_len > 1 and k_len > 1
                for q_len, k_len in zip(self.seqlens_q, self.seqlens_k, strict=True)
            )
        )
        return Exact(
            atol=atol, rtol=rtol, controls=(zeroed_input(0, "query-zeroed"),) if control else ()
        )


class GQAVarlenScaledWorkload(GQAVarlenFwdWorkload):
    """Packed varlen GQA through the op's optional inputs: FP8 scales, fused RoPE.

    An FP8 ``dtype`` adds the per-request, per-KV-head scales and needs a 16-bit
    ``out_dtype``; a ``rotary_dim`` adds the RoPE tables. ``gen_inputs`` emits the
    ten tensor slots the op declares, in signature order, with ``None`` where the
    call omits one.
    """

    def __init__(
        self,
        batch: int,
        seqlens_q: list[int],
        seqlens_k: list[int],
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        wl: int,
        wr: int,
        dtype: torch.dtype,
        sm_scale: float | None = None,
        softcap: float | None = None,
        out_dtype: torch.dtype | None = None,
        rotary_dim: int | None = None,
        rope_layout: str = "neox",
    ) -> None:
        super().__init__(
            batch,
            seqlens_q,
            seqlens_k,
            heads,
            heads_kv,
            dim,
            is_causal,
            wl,
            wr,
            dtype,
            sm_scale=sm_scale,
            softcap=softcap,
        )
        self.out_dtype = dtype if out_dtype is None else out_dtype
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        shapes = (
            (total_q, self.heads, self.dim),
            (total_k, self.heads_kv, self.dim),
            (total_k, self.heads_kv, self.dim),
        )
        if self.dtype == torch.float8_e4m3fn:
            # The score spread in standard deviations is the draw's variance: the head
            # dimension cancels against the default sm_scale. At 0.2 the spread is 0.04,
            # the softmax is uniform to within 4%, and a kernel that never reads its
            # query still agrees with the reference. At 2.0 the spread is 4, which
            # separates a query-blind kernel by 38x to 58x on these rows. A wider draw
            # separates them less: a sharper softmax amplifies the e4m3 score error
            # faster than it sharpens the weights.
            q, k = (
                (torch.randn(shape, device=run_device()) * 2.0).to(self.dtype)
                for shape in shapes[:2]
            )
            v = (torch.randn(shapes[2], device=run_device()) * 0.2).to(self.dtype)
            # Not one: a kernel that never reads a scale must not agree.
            scales = tuple(
                torch.rand(self.batch, self.heads_kv, device=run_device(), dtype=torch.float32)
                * 0.5
                + 0.75
                for _ in range(3)
            )
        else:
            q, k, v = (torch.randn(s, device=run_device(), dtype=self.dtype) for s in shapes)
            scales = (None, None, None)

        rope_cos = rope_sin = None
        if self.rotary_dim is not None:
            angles = (
                torch.randn(self.max_seqlen_kv, self.rotary_dim // 2, device=run_device()) * 0.1
            )
            rope_cos = angles.cos().to(self.out_dtype)
            rope_sin = angles.sin().to(self.out_dtype)
        return (
            q,
            k,
            v,
            make_cu_seqlens(self.seqlens_q),
            make_cu_seqlens(self.seqlens_k),
            *scales,
            rope_cos,
            rope_sin,
        )

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dequantize, rotate, then attend with the inherited per-request reference."""
        groups = self.heads // self.heads_kv
        q_ref, k_ref, v_ref = (t.float() for t in (q, k, v))
        if rope_cos is not None:
            assert rope_sin is not None and self.rotary_dim is not None
            q_positions = torch.cat(
                [
                    torch.arange(kv_len - q_len, kv_len, device=q.device)
                    for q_len, kv_len in zip(self.seqlens_q, self.seqlens_k, strict=True)
                ]
            )
            k_positions = torch.cat(
                [torch.arange(kv_len, device=q.device) for kv_len in self.seqlens_k]
            )
            rope = {"rotary_dim": self.rotary_dim, "layout": self.rope_layout}
            q_ref = apply_packed_rope(q_ref, q_positions, rope_cos, rope_sin, **rope)
            k_ref = apply_packed_rope(k_ref, k_positions, rope_cos, rope_sin, **rope)
        if q_scale is not None:
            assert k_scale is not None and v_scale is not None
            # Formed in FP32, as the kernel's accumulator does. A query head takes
            # the scale of the KV head it attends to, and each request its own row.
            q_rows = torch.repeat_interleave(
                torch.arange(self.batch, device=q.device),
                torch.as_tensor(self.seqlens_q, device=q.device),
            )
            k_rows = torch.repeat_interleave(
                torch.arange(self.batch, device=q.device),
                torch.as_tensor(self.seqlens_k, device=q.device),
            )
            q_ref = q_ref * q_scale.repeat_interleave(groups, dim=1)[q_rows].unsqueeze(-1)
            k_ref = k_ref * k_scale[k_rows].unsqueeze(-1)
            v_ref = v_ref * v_scale[k_rows].unsqueeze(-1)
        return super().ref_program(
            q_ref.to(self.out_dtype),
            k_ref.to(self.out_dtype),
            v_ref.to(self.out_dtype),
            cu_seqlens_q,
            cu_seqlens_kv,
        )


class GQASlidingWindowVarlenFwdWorkload(GQAVarlenFwdWorkload):
    """Compatibility name for the superseded sliding-window public Op."""


class GQAVarlenCall(CallWorkload, GQAVarlenFwdWorkload):
    """A manifest call of GQAVarlenFwdOp, over the request lengths its offsets carry."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.indices, call.params
        q_lens = _segments(call.metadata_values("cu_seqlens_q"))
        GQAVarlenFwdWorkload.__init__(
            self,
            len(q_lens),
            q_lens,
            _segments(call.metadata_values("cu_seqlens_kv")),
            ix["H"],
            ix["H_kv"],
            ix["D"],
            params["is_causal"],
            params.get("window_size_left", -1),
            params.get("window_size_right", -1),
            _dtype(call, "q"),
            sm_scale=params.get("sm_scale"),
            softcap=params.get("softcap"),
            pos_encoding_mode=params.get("pos_encoding_mode", "none"),
            rotary_dim=ix.get("R") if params.get("pos_encoding_mode") == "rope" else None,
            rope_layout=params.get("rope_layout", "neox"),
        )

    gen_inputs = GQAVarlenFwdWorkload.gen_inputs


class GQAVarlenScaledCall(CallWorkload, GQAVarlenScaledWorkload):
    """A manifest call of GQAVarlenFwdOp passing FP8 scales or RoPE tables.

    FP8 values stay inside the format's range and the scales near one; the tables are
    rotations.
    """

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.indices, call.params
        q_lens = _segments(call.metadata_values("cu_seqlens_q"))
        out_dtype = params["out_dtype"]
        rope = params["pos_encoding_mode"] == "rope"
        GQAVarlenScaledWorkload.__init__(
            self,
            len(q_lens),
            q_lens,
            _segments(call.metadata_values("cu_seqlens_kv")),
            ix["H"],
            ix["H_kv"],
            ix["D"],
            params["is_causal"],
            params.get("window_size_left", -1),
            params.get("window_size_right", -1),
            _dtype(call, "q"),
            sm_scale=params.get("sm_scale"),
            softcap=params.get("softcap"),
            out_dtype=None if out_dtype is None else getattr(torch, out_dtype),
            rotary_dim=ix["R"] if rope else None,
            rope_layout=params["rope_layout"],
        )

    gen_inputs = GQAVarlenScaledWorkload.gen_inputs
