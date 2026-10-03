import torch

from workloads.attention.gqa.call_metadata import _dtype
from workloads.attention.gqa.rope import apply_dense_rope
from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "GroupedQueryAttentionDenseDecodeCall",
    "GroupedQueryAttentionDenseDecodeWorkload",
    "GroupedQueryAttentionDensePrefillCall",
    "GroupedQueryAttentionDensePrefillWorkload",
    "dense_gqa_ref",
]


def dense_gqa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    is_causal: bool,
    sm_scale: float | None = None,
    softcap: float | None = None,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    """Grouped-query attention over dense BSHD tensors, accumulated in FP32.

    Query row ``i`` sits at KV position ``i + seq_len_kv - seq_len_q``, which
    the causal flag and the window bounds are read against.
    """
    batch, seq_len_q, _, dim = q.shape
    seq_len_kv = k.shape[1]
    groups = heads // heads_kv
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scale = dim**-0.5 if sm_scale is None else sm_scale
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * scale
    if softcap is not None and softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    offset = seq_len_kv - seq_len_q
    q_pos = torch.arange(seq_len_q, device=q.device)[:, None] + offset
    k_pos = torch.arange(seq_len_kv, device=q.device)[None, :]
    mask = torch.ones((seq_len_q, seq_len_kv), device=q.device, dtype=torch.bool)
    if is_causal:
        mask &= k_pos <= q_pos
    if window_size_left >= 0:
        mask &= k_pos >= q_pos - window_size_left
    if window_size_right >= 0:
        mask &= k_pos <= q_pos + window_size_right
    if is_causal or window_size_left >= 0 or window_size_right >= 0:
        scores = scores.masked_fill(~mask.view(1, 1, seq_len_q, seq_len_kv), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    output = torch.matmul(probs, v_bhsd)
    assert output.shape == (batch, heads, seq_len_q, dim)
    return output.transpose(1, 2).to(q.dtype).contiguous()


class GroupedQueryAttentionDenseDecodeWorkload(WorkloadBase):
    """Single-token decode over a contiguous BSHD KV cache."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_kv: int,
        dim: int,
        dtype: torch.dtype,
        sm_scale: float | None = None,
        softcap: float | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = 0.0 if softcap is None else softcap

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(self.batch, 1, self.heads, self.dim, device=run_device(), dtype=self.dtype)
        k = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim,
            device=run_device(),
            dtype=self.dtype,
        )
        v = torch.randn_like(k)
        return q, k, v

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        groups = self.heads // self.heads_kv
        q_bhsd = q.transpose(1, 2).float()
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * self.sm_scale
        if self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        probs = torch.softmax(scores, dim=-1)
        return torch.matmul(probs, v_bhsd).transpose(1, 2).to(q.dtype).contiguous()


class GroupedQueryAttentionDensePrefillWorkload(WorkloadBase):
    """Dense prefill over contiguous BSHD tensors, with the op's optional inputs.

    An FP8 ``dtype`` adds the per-KV-head scales and needs a 16-bit
    ``out_dtype``; a ``rotary_dim`` adds the RoPE tables. ``gen_inputs`` emits
    the eight tensor slots the op declares, in signature order, with ``None``
    where the call omits one.
    """

    def __init__(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        heads: int,
        heads_kv: int,
        dim: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype | None = None,
        is_causal: bool = True,
        sm_scale: float | None = None,
        softcap: float | None = None,
        rotary_dim: int | None = None,
        rope_layout: str = "neox",
    ) -> None:
        self.batch = batch
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dtype = dtype
        self.out_dtype = dtype if out_dtype is None else out_dtype
        self.is_causal = is_causal
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = 0.0 if softcap is None else softcap
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        shapes = (
            (self.batch, self.seq_len_q, self.heads, self.dim),
            (self.batch, self.seq_len_kv, self.heads_kv, self.dim),
            (self.batch, self.seq_len_kv, self.heads_kv, self.dim),
        )
        if self.dtype == torch.float8_e4m3fn:
            # FP8 saturates near 448; 0.2 keeps the products in range.
            q, k, v = ((torch.randn(s, device=run_device()) * 0.2).to(self.dtype) for s in shapes)
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
            angles = torch.randn(self.seq_len_kv, self.rotary_dim // 2, device=run_device()) * 0.1
            rope_cos = angles.cos().to(self.out_dtype)
            rope_sin = angles.sin().to(self.out_dtype)
        return (q, k, v, *scales, rope_cos, rope_sin)

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        groups = self.heads // self.heads_kv
        q_ref, k_ref, v_ref = (t.to(self.out_dtype) for t in (q, k, v))
        if q_scale is not None:
            assert k_scale is not None and v_scale is not None
            # Formed in FP32, as the kernel's accumulator does. A query head
            # takes the scale of the KV head it attends to.
            per_query_head = q_scale.repeat_interleave(groups, dim=1)
            q_ref = (q_ref.float() * per_query_head.view(self.batch, 1, self.heads, 1)).to(
                self.out_dtype
            )
            k_ref = (k_ref.float() * k_scale.view(self.batch, 1, self.heads_kv, 1)).to(
                self.out_dtype
            )
            v_ref = (v_ref.float() * v_scale.view(self.batch, 1, self.heads_kv, 1)).to(
                self.out_dtype
            )
        if rope_cos is not None:
            assert rope_sin is not None and self.rotary_dim is not None
            offset = self.seq_len_kv - self.seq_len_q
            q_positions = torch.arange(offset, self.seq_len_kv, device=q.device)
            k_positions = torch.arange(self.seq_len_kv, device=q.device)
            q_ref = apply_dense_rope(
                q_ref,
                q_positions,
                rope_cos,
                rope_sin,
                rotary_dim=self.rotary_dim,
                layout=self.rope_layout,
            )
            k_ref = apply_dense_rope(
                k_ref,
                k_positions,
                rope_cos,
                rope_sin,
                rotary_dim=self.rotary_dim,
                layout=self.rope_layout,
            )
        return dense_gqa_ref(
            q_ref,
            k_ref,
            v_ref,
            heads=self.heads,
            heads_kv=self.heads_kv,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
        )


class GroupedQueryAttentionDenseDecodeCall(CallWorkload, GroupedQueryAttentionDenseDecodeWorkload):
    """A manifest call of GroupedQueryAttentionDenseFwdOp with one query token."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix = call.ix
        GroupedQueryAttentionDenseDecodeWorkload.__init__(
            self,
            ix["B"],
            ix["H"],
            ix["H_kv"],
            ix["S_kv"],
            ix["D"],
            _dtype(call, "q"),
            sm_scale=call.params["sm_scale"],
            softcap=call.params["softcap"],
        )

    gen_inputs = GroupedQueryAttentionDenseDecodeWorkload.gen_inputs


class GroupedQueryAttentionDensePrefillCall(
    CallWorkload, GroupedQueryAttentionDensePrefillWorkload
):
    """A manifest call of GroupedQueryAttentionDenseFwdOp passing FP8 scales or RoPE tables.

    FP8 values stay inside the format's range and the scales near one; the tables are
    rotations.
    """

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        out_dtype = params["out_dtype"]
        rope = params["pos_encoding_mode"] == "rope"
        GroupedQueryAttentionDensePrefillWorkload.__init__(
            self,
            ix["B"],
            ix["S_q"],
            ix["S_kv"],
            ix["H"],
            ix["H_kv"],
            ix["D"],
            _dtype(call, "q"),
            out_dtype=None if out_dtype is None else getattr(torch, out_dtype),
            is_causal=params["is_causal"],
            sm_scale=params["sm_scale"],
            softcap=params["softcap"],
            rotary_dim=ix["R"] if rope else None,
            rope_layout=params["rope_layout"],
        )

    gen_inputs = GroupedQueryAttentionDensePrefillWorkload.gen_inputs
