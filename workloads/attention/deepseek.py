"""Workload definitions for the DeepSeek attention ops."""

from typing import Optional, Union

import torch
from einops import rearrange, repeat

from workloads.nsa_utils import prepare_chunk_offsets, prepare_token_indices
from workloads.workload_base import WorkloadBase


class NsaFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, c_seq_len: int, dim: int, is_causal: bool,
                 scale: float, block_size: int, groups: int, selected_blocks: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.c_seq_len = c_seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        self.head_kv = self.heads // self.groups

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        possible_split_points = torch.arange(16, self.c_seq_len)
        num_splits = self.batch - 1
        offsets = (
            torch.cat(
                [
                    torch.tensor([0], dtype=torch.long),
                    possible_split_points[torch.randperm(len(possible_split_points))[:num_splits]],
                    torch.tensor([self.c_seq_len], dtype=torch.long),
                ],
                0,
            ).cuda().sort()[0])

        perm_q = torch.randperm(self.c_seq_len, device="cuda")
        perm_k = torch.randperm(self.c_seq_len, device="cuda")
        perm_v = torch.randperm(self.c_seq_len, device="cuda")
        q = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype,
                           device="cuda")[perm_q].view(1, self.c_seq_len, 1, 1).expand(
                               1, self.c_seq_len, self.heads,
                               self.dim).clone().requires_grad_(True))
        k = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype,
                           device="cuda")[perm_k].view(1, self.c_seq_len, 1, 1).expand(
                               1, self.c_seq_len, self.head_kv,
                               self.dim).clone().requires_grad_(True))
        v = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype,
                           device="cuda")[perm_v].view(1, self.c_seq_len, 1, 1).expand(
                               1, self.c_seq_len, self.head_kv,
                               self.dim).clone().requires_grad_(True))
        self.o_slc = torch.empty((self.batch, self.c_seq_len, self.heads, self.dim),
                                 dtype=self.dtype,
                                 device="cuda")
        self.lse_slc = torch.empty((self.batch, self.c_seq_len, self.heads, self.dim),
                                   dtype=torch.float,
                                   device="cuda")

        self.g_slc = torch.ones((self.batch, self.c_seq_len, self.heads),
                                dtype=self.dtype,
                                device="cuda").requires_grad_(True)
        self.g_swa = torch.ones((self.batch, self.c_seq_len, self.heads),
                                dtype=self.dtype,
                                device="cuda").requires_grad_(True)

        token_indices = prepare_token_indices(offsets)
        token_indices_list = token_indices.tolist()
        block_indices = torch.full(
            (1, self.c_seq_len, self.head_kv, self.selected_blocks),
            self.c_seq_len,
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.c_seq_len):
            _, t = token_indices_list[i]
            chunks = max(1, (t + self.block_size - 1) // self.block_size)
            for h in range(self.head_kv):
                i_i = torch.randperm(chunks)[:self.selected_blocks]
                block_indices[0, i, h, :len(i_i)] = i_i
        block_indices = block_indices.sort(-1)[0]
        block_counts = torch.randint(
            1,
            self.selected_blocks + 1,
            (1, self.c_seq_len, self.head_kv),
            dtype=torch.int32,
            device="cuda",
        )
        return (
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            block_indices.squeeze(0),
            block_counts.squeeze(0),
            offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )

    def naive_nsa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_slc: torch.Tensor,
        g_swa: torch.Tensor,
        block_indices: torch.LongTensor,
        block_counts: Optional[Union[torch.LongTensor, int]] = None,
        block_size: int = 64,
        window_size: int = 0,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        head_first: bool = False,
    ) -> torch.Tensor:

        if scale is None:
            scale = k.shape[-1]**-0.5
        if cu_seqlens is not None:
            assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
            if head_first:
                raise RuntimeError(
                    "Sequences with variable lengths are not supported for head-first mode")
        if head_first:
            q, k, v, block_indices = (
                rearrange(x, "b h t d -> b t h d") for x in (q, k, v, block_indices))
            g_slc, g_swa = (rearrange(x, "b h t -> b t h") for x in (g_slc, g_swa))
            if isinstance(block_counts, torch.Tensor):
                block_counts = rearrange(block_counts, "b h t -> b t h")

        dtype = q.dtype
        g = q.shape[2] // k.shape[2]
        bs = block_size
        s = block_indices.shape[-1]
        k, v, block_indices = (
            repeat(x, "b t h d -> b t (h g) d", g=g) for x in (k, v, block_indices))
        if isinstance(block_counts, torch.Tensor):
            block_counts = repeat(block_counts, "b t h -> b t (h g)", g=g)
        c = torch.arange(s).repeat_interleave(bs).unsqueeze(1).expand(-1, q.shape[2]).to(q.device)
        q, k, v = (x.float() for x in (q, k, v))

        o_slc = torch.zeros_like(v)
        o_swa = torch.zeros_like(v) if window_size > 0 else None
        varlen = True
        if cu_seqlens is None:
            varlen = False
            b, t = q.shape[:2]
            cu_seqlens = torch.cat(
                [block_indices.new_tensor(range(0, b * t, t)),
                 block_indices.new_tensor([b * t])])

        for i in range(len(cu_seqlens) - 1):
            if not varlen:
                q_b, k_b, v_b = q[i], k[i], v[i]
                g_slc_b, g_swa_b, i_b = g_slc[i], g_swa[i], block_indices[i]
                s_b = block_counts[i] if isinstance(block_counts, torch.Tensor) else block_counts
            else:
                t = cu_seqlens[i + 1] - cu_seqlens[i]
                q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = (
                    x[0][cu_seqlens[i]:cu_seqlens[i + 1]]
                    for x in (q, k, v, g_slc, g_swa, block_indices))
                s_b = (
                    block_counts[0][cu_seqlens[i]:cu_seqlens[i + 1]] if isinstance(
                        block_counts, torch.Tensor) else block_counts)

            i_b = i_b.unsqueeze(-1) * bs + i_b.new_tensor(range(bs))
            # [t, s*bs, hq]
            i_b = i_b.view(t, block_indices.shape[2], -1).transpose(1, 2)
            for i_q in range(t):
                # [hq, d]
                q_i = q_b[i_q] * scale
                # [hq]
                g_slc_i = g_slc_b[i_q]
                # [hq]
                g_swa_i = g_swa_b[i_q]
                i_i = i_b[i_q]
                s_i = s_b[i_q] if isinstance(block_counts, torch.Tensor) else s_b
                k_i_slc, v_i_slc = (
                    x.gather(0,
                             i_i.clamp(0, t - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1]))
                    for x in (k_b, v_b))
                # [s*bs, hq]
                attn_slc = (
                    torch.einsum("h d, n h d -> n h", q_i, k_i_slc).masked_fill(
                        torch.logical_or(i_i < 0, i_i > i_q)
                        | (c >= s_i if block_counts is not None else False),
                        float("-inf")).softmax(0))
                if not varlen:
                    o_slc[i, i_q] = torch.einsum("n h, n h v -> h v", attn_slc,
                                                 v_i_slc) * g_slc_i.unsqueeze(-1)
                else:
                    o_slc[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn_slc,
                                                                 v_i_slc) * g_slc_i.unsqueeze(-1)
                if window_size > 0:
                    k_i_swa, v_i_swa = (
                        x[max(0, i_q - window_size + 1):i_q + 1] for x in (k_b, v_b))
                    attn_swa = torch.einsum("h d, n h d -> n h", q_i, k_i_swa).softmax(0)
                    if not varlen:
                        o_swa[i, i_q] = torch.einsum("n h, n h v -> h v", attn_swa,
                                                     v_i_swa) * g_swa_i.unsqueeze(-1)
                    else:
                        o_swa[0][cu_seqlens[i] + i_q] = torch.einsum(
                            "n h, n h v -> h v", attn_swa, v_i_swa) * g_swa_i.unsqueeze(-1)

        if head_first:
            o_slc = rearrange(o_slc, "b t h d -> b h t d")
            o_swa = rearrange(o_swa, "b t h d -> b h t d")

        return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


class NsaCmpFwdTest(WorkloadBase):

    def __init__(self, seq_num: int, c_seq_len: int, heads: int, dim_k: int, dim_v: int,
                 group: int, scale: float, bc: int, bs: int, bk: int, bv: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.group = group
        self.scale = scale
        self.bc = bc
        self.bs = bs
        self.bk = bk
        self.bv = bv
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        self.head_kv = self.heads // self.group
        # chunk_num is computed during gen_inputs and stored for later use
        self.chunk_num = None

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        valid_range = self.c_seq_len - self.bs
        rand_indices = torch.randperm(valid_range)[:self.seq_num - 1]
        offsets = torch.cat([
            torch.tensor([0]),
            torch.arange(self.bs, self.c_seq_len)[rand_indices],
            torch.tensor([self.c_seq_len])
        ], 0).cuda().sort()[0].to(torch.int32)

        chunk_offsets = prepare_chunk_offsets(offsets, self.bs).to(torch.int32)
        token_indices = prepare_token_indices(offsets).to(torch.int32)
        chunk_num = chunk_offsets[-1].item()

        # float16, data Tie-breaking
        q = torch.randn((self.c_seq_len, self.heads, self.dim_k), dtype=self.dtype, device="cuda")
        k = torch.randn((chunk_num, self.head_kv, self.dim_k), dtype=self.dtype, device="cuda")
        v = torch.randn((chunk_num, self.head_kv, self.dim_v), dtype=self.dtype, device="cuda")

        self.chunk_num = chunk_offsets[-1].item()
        return (
            q,
            k,
            v,
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )


class NsaTopkTest(WorkloadBase):

    def __init__(self, seq_num: int, c_seq_len: int, heads: int, dim: int, group: int,
                 scale: float, selected_block_num: int, bc: int, bs: int, bk: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.group = group
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bc = bc
        self.bs = bs
        self.bk = bk
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        self.head_kv = self.heads // self.group
        # chunk_num is computed during gen_inputs and stored for later use
        self.chunk_num = None

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        possible_split_points = torch.arange(16, self.c_seq_len)
        num_splits = self.seq_num - 1
        offsets = (
            torch.cat(
                [
                    torch.tensor([0], dtype=torch.long),
                    possible_split_points[torch.randperm(len(possible_split_points))[:num_splits]],
                    torch.tensor([self.c_seq_len], dtype=torch.long),
                ],
                0,
            ).cuda().sort()[0])

        chunk_offsets = prepare_chunk_offsets(offsets, self.bs)
        token_indices = prepare_token_indices(offsets)
        chunk_num = chunk_offsets[-1].item()

        # float16, data Tie-breaking
        q = torch.randn(
            (self.c_seq_len, self.heads, self.dim), dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn((chunk_num, self.head_kv, self.dim), dtype=self.dtype, device="cuda") * 0.1

        q.requires_grad_(True)
        k.requires_grad_(True)

        lse = torch.zeros((self.c_seq_len, self.heads), dtype=self.dtype, device="cuda")

        self.chunk_num = chunk_offsets[-1].item()
        return (
            q,
            k,
            lse,
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )


class MlaDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                 dim_pe: int, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device='cuda', dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.dim_pe, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim,
            device='cuda',
            dtype=self.dtype)
        K_pe = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim_pe,
            device='cuda',
            dtype=self.dtype)
        return Q, Q_pe, K, K_pe


class DsaDecodeTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seq_len: int, seq_len_kv: int, dim: int,
                 dim_tail: int, topk: int, stride_kv: int, heads_kv: int, q_start_index_s: int,
                 sm_scale: float = None, is_causal: bool = True,
                 dtype: torch.dtype = torch.float16) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.heads_kv = heads_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim + self.dim_tail,
            device='cuda',
            dtype=self.dtype)
        indices = torch.full((self.batch, self.seq_len, self.heads_kv, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device='cuda')
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.heads_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv))[:self.topk]
                    indices[b, t, h, :len(i_i)] = i_i
        return q, kv, indices
