"""Workload definitions for the GQA attention ops."""

import math

import torch

from tileops.ops import GroupedQueryAttentionPrefillDenseFwdOp
from workloads.workload_base import WorkloadBase


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

        fwd_op = GroupedQueryAttentionPrefillDenseFwdOp(
            self.batch, self.heads, self.heads_kv, self.seq_len, self.dim, self.is_causal
        )
        with torch.no_grad():
            o = fwd_op(q, k, v)
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads_kv,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse


class GroupedQueryAttentionFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: float | None = None,
        softcap: float | None = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        seq_len_kv: int | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.seq_len_kv = seq_len if seq_len_kv is None else seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        k = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        v = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        ).contiguous()
        return q, k, v


class GroupedQueryAttentionDecodeWorkload(WorkloadBase):
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
        Q = torch.randn(self.batch, self.heads, self.dim, device="cuda", dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        )
        V = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        )
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        groups = self.heads // self.heads_kv
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scores = torch.matmul(q_bhsd.float(), k_bhsd.transpose(-2, -1)) * self.sm_scale
        if self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        probs = torch.softmax(scores, dim=-1)
        output_bhsd = torch.matmul(probs, v_bhsd)
        return output_bhsd.transpose(1, 2).squeeze(1).to(q.dtype).contiguous()


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
        block_table = (
            torch.arange(num_pages, dtype=torch.int32, device="cuda")
            .unsqueeze(0)
            .expand(self.batch, -1)
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
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
        sm_scale: float | None = None,
        softcap: float | None = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.q_lens = q_lens
        self.kv_lens = kv_lens
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.output_dtype = output_dtype or dtype

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

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        fp8 = getattr(torch, "float8_e4m3fn", None)
        source_dtype = torch.float16 if self.dtype == fp8 else self.dtype
        q = torch.randn(
            self.total_q, self.heads, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        k = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        v = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv = torch.tensor(
            [0] + torch.tensor(self.kv_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
        )
        scale = torch.full(
            (self.batch, self.heads_kv),
            0.05 if self.dtype == fp8 else 1.0,
            dtype=torch.float32,
            device="cuda",
        )
        if self.dtype == fp8:
            groups = self.heads // self.heads_kv
            q_parts, k_parts, v_parts = [], [], []
            for b in range(self.batch):
                q_start = int(cu_seqlens_q[b].item())
                q_end = int(cu_seqlens_q[b + 1].item())
                kv_start = int(cu_seqlens_kv[b].item())
                kv_end = int(cu_seqlens_kv[b + 1].item())
                q_head_scale = scale[b].repeat_interleave(groups).view(1, self.heads, 1)
                kv_head_scale = scale[b].view(1, self.heads_kv, 1)
                q_parts.append((q[q_start:q_end].float() / q_head_scale).to(fp8))
                k_parts.append((k[kv_start:kv_end].float() / kv_head_scale).to(fp8))
                v_parts.append((v[kv_start:kv_end].float() / kv_head_scale).to(fp8))
            q = torch.cat(q_parts).contiguous()
            k = torch.cat(k_parts).contiguous()
            v = torch.cat(v_parts).contiguous()
        return (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            scale,
            scale.clone(),
            scale.clone(),
        )


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
        sm_scale: float | None = None,
        softcap: float | None = None,
        window_size_left: int = -1,
        window_size_right: int = -1,
        append_kv: bool = True,
        cache_dtype: torch.dtype | None = None,
        output_dtype: torch.dtype | None = None,
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
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.output_dtype = output_dtype or dtype
        self.append_kv = append_kv
        self.cache_dtype = cache_dtype or dtype

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
        fp8 = getattr(torch, "float8_e4m3fn", None)
        source_dtype = torch.float16 if self.dtype == fp8 else self.dtype
        cache_source_dtype = torch.float16 if self.cache_dtype == fp8 else self.cache_dtype
        q = torch.randn(
            self.total_q, self.heads, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        k_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        v_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device="cuda", dtype=source_dtype
        ).contiguous()
        physical_tokens = self.batch * self.max_pages_per_req * self.page_size
        k_pages = torch.randn(
            physical_tokens, self.heads_kv, self.dim, device="cuda", dtype=cache_source_dtype
        ).contiguous()
        v_pages = torch.randn_like(k_pages)
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda"
        )
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device="cuda")
        block_table = (
            torch.arange(self.batch * self.max_pages_per_req, dtype=torch.int32, device="cuda")
            .reshape(self.batch, self.max_pages_per_req)
            .contiguous()
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
            self.max_seqlen_q,
        )
