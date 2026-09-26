"""Workload definitions for the GQA attention ops."""

import math
from itertools import accumulate

import torch
import torch.nn.functional as F

from workloads.attention.paged import make_fragmented_block_table
from workloads.workload_base import CallWorkload, WorkloadBase


def make_cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Exclusive prefix sum of *lengths*, the packed-varlen offset vector."""
    return torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)


def _compute_gqa_square_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    heads: int,
    heads_kv: int,
    dim: int,
    is_causal: bool,
) -> torch.Tensor:
    groups = heads // heads_kv
    seq_len = q.shape[1]
    q_bhsd = q.transpose(1, 2).float()
    k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-2, -1)) * (dim**-0.5)
    if is_causal:
        pos = torch.arange(seq_len, device=q.device)
        mask = pos[None, :] <= pos[:, None]
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), float("-inf"))
    return torch.logsumexp(scores, dim=-1) * math.log2(math.e)


def apply_dense_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
    layout: str,
) -> torch.Tensor:
    """Rotate the first *rotary_dim* channels of a BSHD tensor at *positions*.

    ``neox`` pairs channel ``i`` with ``i + rotary_dim // 2``, ``interleaved``
    pairs adjacent channels, and channels past *rotary_dim* pass through.
    """
    half = rotary_dim // 2
    x_rot = x[..., :rotary_dim].float()
    c = cos[positions].view(1, x.shape[1], 1, half).float()
    s = sin[positions].view(1, x.shape[1], 1, half).float()
    if layout == "neox":
        x0, x1 = x_rot[..., :half], x_rot[..., half:]
    else:
        x0, x1 = x_rot[..., 0::2], x_rot[..., 1::2]
    y0, y1 = x0 * c - x1 * s, x1 * c + x0 * s
    rotated = (
        torch.cat((y0, y1), dim=-1)
        if layout == "neox"
        else torch.stack((y0, y1), dim=-1).flatten(-2)
    )
    return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1).contiguous()


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


class GroupedQueryAttentionBwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device="cuda"
        )

        with torch.no_grad():
            o = (
                F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=self.is_causal,
                    enable_gqa=True,
                )
                .transpose(1, 2)
                .contiguous()
            )
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads_kv,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse


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
        q = torch.randn(self.batch, 1, self.heads, self.dim, device="cuda", dtype=self.dtype)
        k = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim,
            device="cuda",
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
            q, k, v = ((torch.randn(s, device="cuda") * 0.2).to(self.dtype) for s in shapes)
            # Not one: a kernel that never reads a scale must not agree.
            scales = tuple(
                torch.rand(self.batch, self.heads_kv, device="cuda", dtype=torch.float32) * 0.5
                + 0.75
                for _ in range(3)
            )
        else:
            q, k, v = (torch.randn(s, device="cuda", dtype=self.dtype) for s in shapes)
            scales = (None, None, None)

        rope_cos = rope_sin = None
        if self.rotary_dim is not None:
            angles = torch.randn(self.seq_len_kv, self.rotary_dim // 2, device="cuda") * 0.1
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


class GroupedQueryAttentionDecodePagedWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        dtype: torch.dtype,
        sm_scale: float | None = None,
        softcap: float | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = 0.0 if softcap is None else softcap

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.randint(
            self.page_size, self.seqlen_kv + 1, (self.batch,), dtype=torch.int32, device="cuda"
        )
        real_seqlen_kv = (real_seqlen_kv // self.page_size) * self.page_size
        real_seqlen_kv[0] = min(real_seqlen_kv[0].item(), self.seqlen_kv)

        q = torch.randn(self.batch, self.heads, self.dim, dtype=self.dtype, device="cuda")
        k = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
        v = torch.randn(self.seqlen_kv, self.heads_kv, self.dim, dtype=self.dtype, device="cuda")
        block_table = make_fragmented_block_table(self.batch, num_pages, num_pages)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table


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
            self.total_q, self.heads, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        k = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        v = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv = torch.tensor(
            [0] + torch.tensor(self.kv_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
        )
        return q, k, v, cu_seqlens_q, cu_seqlens_kv


class GQAPrefillPagedWithKVCacheFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        q_lens: list[int],
        cache_lens: list[int],
        page_size: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        fuse_rope: bool = False,
        rotary_dim: int | None = None,
        softcap: float | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.q_lens = q_lens
        self.cache_lens = cache_lens
        self.page_size = page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.fuse_rope = fuse_rope
        self.rotary_dim = rotary_dim
        self.softcap = softcap

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.q_lens)

    @property
    def max_total_len(self) -> int:
        return max(cache + q for cache, q in zip(self.cache_lens, self.q_lens, strict=True))

    @property
    def max_pages_per_req(self) -> int:
        return (self.max_total_len + self.page_size - 1) // self.page_size

    def gen_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        q = torch.randn(
            self.total_q, self.heads, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        k_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        v_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        physical_tokens = self.batch * self.max_pages_per_req * self.page_size
        k_pages = torch.randn(
            physical_tokens, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        v_pages = torch.randn_like(k_pages)
        cu_seqlens_q = make_cu_seqlens(self.q_lens)
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device="cuda")
        block_table = make_fragmented_block_table(
            self.batch, self.max_pages_per_req, self.batch * self.max_pages_per_req
        )
        return (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )


class GroupedQueryAttentionVarlenFwdWorkload(WorkloadBase):
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

    @property
    def max_seqlen_q(self) -> int:
        return max(self.seqlens_q)

    @property
    def max_seqlen_kv(self) -> int:
        return max(self.seqlens_k)

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        total_q = sum(self.seqlens_q)
        total_k = sum(self.seqlens_k)
        q = torch.randn(total_q, self.heads, self.dim, dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(total_k, self.heads_kv, self.dim, dtype=self.dtype, device="cuda") * 0.1

        cu_seqlens_q = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_q), 0).tolist()),
            dtype=torch.int32,
            device="cuda",
        )
        cu_seqlens_k = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(self.seqlens_k), 0).tolist()),
            dtype=torch.int32,
            device="cuda",
        )
        return q, k, v, cu_seqlens_q, cu_seqlens_k

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Canonical materialized reference for regular and windowed Varlen GQA."""
        groups = self.heads // self.heads_kv
        scale = self.dim**-0.5 if self.sm_scale is None else self.sm_scale
        outputs = []
        for request in range(self.batch):
            q_start = int(cu_seqlens_q[request].item())
            q_end = int(cu_seqlens_q[request + 1].item())
            kv_start = int(cu_seqlens_kv[request].item())
            kv_end = int(cu_seqlens_kv[request + 1].item())
            q_i = q[q_start:q_end].transpose(0, 1).float()
            k_i = k[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            v_i = v[kv_start:kv_end].repeat_interleave(groups, dim=1).permute(1, 0, 2).float()
            q_len = q_end - q_start
            kv_len = kv_end - kv_start
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


class GroupedQueryAttentionSlidingWindowVarlenFwdWorkload(GroupedQueryAttentionVarlenFwdWorkload):
    """Compatibility name for the superseded sliding-window public Op."""


def _dtype(call, tensor: str) -> torch.dtype:
    return getattr(torch, call.tensors[tensor][1])


def _segments(bounds: list[int]) -> list[int]:
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]


class GroupedQueryAttentionBwdCall(CallWorkload, GroupedQueryAttentionBwdWorkload):
    """A manifest call of GroupedQueryAttentionBwdOp; ``o`` and ``lse`` are the forward's."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix = call.ix
        GroupedQueryAttentionBwdWorkload.__init__(
            self, ix["B"], ix["H"], ix["H_kv"], ix["S"], ix["D"], ix["is_causal"], _dtype(call, "q")
        )

    gen_inputs = GroupedQueryAttentionBwdWorkload.gen_inputs


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


class GroupedQueryAttentionVarlenCall(CallWorkload, GroupedQueryAttentionVarlenFwdWorkload):
    """A manifest call of a packed GQA op, over the request lengths its offsets carry."""

    def __init__(self, call, cu_kv: str = "cu_seqlens_kv") -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        q_lens = _segments(call.values("cu_seqlens_q"))
        GroupedQueryAttentionVarlenFwdWorkload.__init__(
            self,
            len(q_lens),
            q_lens,
            _segments(call.values(cu_kv)),
            ix["H"],
            ix["H_kv"],
            ix["D"],
            params["is_causal"],
            params.get("window_size_left", -1),
            params.get("window_size_right", -1),
            _dtype(call, "q"),
            sm_scale=params.get("sm_scale"),
            softcap=params.get("softcap"),
        )

    gen_inputs = GroupedQueryAttentionVarlenFwdWorkload.gen_inputs


class GQAPrefillPagedWithKVCacheFwdCall(CallWorkload, GQAPrefillPagedWithKVCacheFwdWorkload):
    """A manifest call of GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp.

    An FP8 cache holds the random pages quantized by scales of 0.01; an unquantized one
    passes unit scales.
    """

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        q_lens = _segments(call.values("cu_seqlens_q"))
        GQAPrefillPagedWithKVCacheFwdWorkload.__init__(
            self,
            len(q_lens),
            ix["H"],
            ix["H_kv"],
            q_lens,
            call.values("cache_seqlens"),
            params["page_size"],
            ix["D"],
            params["is_causal"],
            _dtype(call, "q"),
            fuse_rope=params["fuse_rope"],
            rotary_dim=params["rotary_dim"],
            softcap=params["softcap"],
        )
        self.cache_dtype = _dtype(call, "k_pages")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table = (
            GQAPrefillPagedWithKVCacheFwdWorkload.gen_inputs(self)
        )
        scale = 1.0
        if self.cache_dtype == torch.float8_e4m3fn:
            scale = 0.01
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            k_pages, v_pages = (
                (t / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
                for t in (k_pages, v_pages)
            )
        k_scale = torch.full((1,), scale, dtype=torch.float32, device=q.device)
        return (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            k_scale.clone(),
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )


class GroupedQueryAttentionDecodePagedCall(CallWorkload, GroupedQueryAttentionDecodePagedWorkload):
    """A manifest call of GroupedQueryAttentionDecodePagedWithKVCacheFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        GroupedQueryAttentionDecodePagedWorkload.__init__(
            self,
            ix["B"],
            ix["H"],
            ix["H_kv"],
            ix["N_kv"],
            ix["D"],
            params["page_size"],
            _dtype(call, "q"),
            sm_scale=params["sm_scale"],
            softcap=params["softcap"],
        )

    gen_inputs = CallWorkload.gen_inputs
