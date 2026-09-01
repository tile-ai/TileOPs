"""Workload definitions for the DeepSeek attention ops."""

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from workloads.nsa_utils import prepare_chunk_offsets, prepare_token_indices
from workloads.workload_base import WorkloadBase


def _packed_offsets(
    seq_lens: "list[int] | None", seq_num: int, c_seq_len: int, min_split: int
) -> torch.Tensor:
    """Request boundaries into a packed sequence of ``c_seq_len`` tokens.

    Explicit ``seq_lens`` make the chunk count deterministic; without them the split
    points are random.
    """
    if seq_lens is not None:
        if sum(seq_lens) != c_seq_len or len(seq_lens) != seq_num:
            raise ValueError(
                f"seq_lens must hold {seq_num} lengths summing to {c_seq_len}, "
                f"got {len(seq_lens)} summing to {sum(seq_lens)}"
            )
        bounds = torch.tensor([0, *seq_lens], dtype=torch.long).cumsum(0)
        return bounds.cuda()
    splits = torch.arange(min_split, c_seq_len)
    return (
        torch.cat(
            [
                torch.tensor([0], dtype=torch.long),
                splits[torch.randperm(len(splits))[: seq_num - 1]],
                torch.tensor([c_seq_len], dtype=torch.long),
            ],
            0,
        )
        .cuda()
        .sort()[0]
    )


class NsaFwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        c_seq_len: int,
        dim: int,
        is_causal: bool,
        scale: float,
        block_size: int,
        groups: int,
        selected_blocks: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        seq_lens: "list[int] | None" = None,
    ) -> None:
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
        self.seq_lens = seq_lens

        self.head_kv = self.heads // self.groups

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        offsets = _packed_offsets(self.seq_lens, self.batch, self.c_seq_len, 16)

        perm_q = torch.randperm(self.c_seq_len, device="cuda")
        perm_k = torch.randperm(self.c_seq_len, device="cuda")
        perm_v = torch.randperm(self.c_seq_len, device="cuda")
        q = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype, device="cuda")[perm_q]
            .view(1, self.c_seq_len, 1, 1)
            .expand(1, self.c_seq_len, self.heads, self.dim)
            .clone()
            .requires_grad_(True)
        )
        k = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype, device="cuda")[perm_k]
            .view(1, self.c_seq_len, 1, 1)
            .expand(1, self.c_seq_len, self.head_kv, self.dim)
            .clone()
            .requires_grad_(True)
        )
        v = (
            torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype, device="cuda")[perm_v]
            .view(1, self.c_seq_len, 1, 1)
            .expand(1, self.c_seq_len, self.head_kv, self.dim)
            .clone()
            .requires_grad_(True)
        )
        self.o_slc = torch.empty(
            (self.batch, self.c_seq_len, self.heads, self.dim), dtype=self.dtype, device="cuda"
        )
        self.lse_slc = torch.empty(
            (self.batch, self.c_seq_len, self.heads, self.dim), dtype=torch.float, device="cuda"
        )

        self.g_slc = torch.ones(
            (self.batch, self.c_seq_len, self.heads), dtype=self.dtype, device="cuda"
        ).requires_grad_(True)
        self.g_swa = torch.ones(
            (self.batch, self.c_seq_len, self.heads), dtype=self.dtype, device="cuda"
        ).requires_grad_(True)

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
                i_i = torch.randperm(chunks)[: self.selected_blocks]
                block_indices[0, i, h, : len(i_i)] = i_i
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

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        offsets: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> torch.Tensor:
        _ = token_indices
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        block_indices = block_indices.unsqueeze(0)
        block_counts = block_counts.unsqueeze(0)
        g_slc = self.g_slc
        g_swa = self.g_swa
        block_size = self.block_size
        window_size = 0
        scale = self.scale
        cu_seqlens = offsets
        head_first = False
        if scale is None:
            scale = k.shape[-1] ** -0.5
        if cu_seqlens is not None:
            assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
            if head_first:
                raise RuntimeError(
                    "Sequences with variable lengths are not supported for head-first mode"
                )
        if head_first:
            q, k, v, block_indices = (
                rearrange(x, "b h t d -> b t h d") for x in (q, k, v, block_indices)
            )
            g_slc, g_swa = (rearrange(x, "b h t -> b t h") for x in (g_slc, g_swa))
            if isinstance(block_counts, torch.Tensor):
                block_counts = rearrange(block_counts, "b h t -> b t h")

        dtype = q.dtype
        g = q.shape[2] // k.shape[2]
        bs = block_size
        s = block_indices.shape[-1]
        k, v, block_indices = (
            repeat(x, "b t h d -> b t (h g) d", g=g) for x in (k, v, block_indices)
        )
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
                [block_indices.new_tensor(range(0, b * t, t)), block_indices.new_tensor([b * t])]
            )

        for i in range(len(cu_seqlens) - 1):
            if not varlen:
                q_b, k_b, v_b = q[i], k[i], v[i]
                g_slc_b, g_swa_b, i_b = g_slc[i], g_swa[i], block_indices[i]
                s_b = block_counts[i] if isinstance(block_counts, torch.Tensor) else block_counts
            else:
                t = cu_seqlens[i + 1] - cu_seqlens[i]
                q_b, k_b, v_b, g_slc_b, g_swa_b, i_b = (
                    x[0][cu_seqlens[i] : cu_seqlens[i + 1]]
                    for x in (q, k, v, g_slc, g_swa, block_indices)
                )
                s_b = (
                    block_counts[0][cu_seqlens[i] : cu_seqlens[i + 1]]
                    if isinstance(block_counts, torch.Tensor)
                    else block_counts
                )

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
                    x.gather(0, i_i.clamp(0, t - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1]))
                    for x in (k_b, v_b)
                )
                # [s*bs, hq]
                attn_slc = (
                    torch.einsum("h d, n h d -> n h", q_i, k_i_slc)
                    .masked_fill(
                        torch.logical_or(i_i < 0, i_i > i_q)
                        | (c >= s_i if block_counts is not None else False),
                        float("-inf"),
                    )
                    .softmax(0)
                )
                if not varlen:
                    o_slc[i, i_q] = torch.einsum(
                        "n h, n h v -> h v", attn_slc, v_i_slc
                    ) * g_slc_i.unsqueeze(-1)
                else:
                    o_slc[0][cu_seqlens[i] + i_q] = torch.einsum(
                        "n h, n h v -> h v", attn_slc, v_i_slc
                    ) * g_slc_i.unsqueeze(-1)
                if window_size > 0:
                    k_i_swa, v_i_swa = (
                        x[max(0, i_q - window_size + 1) : i_q + 1] for x in (k_b, v_b)
                    )
                    attn_swa = torch.einsum("h d, n h d -> n h", q_i, k_i_swa).softmax(0)
                    if not varlen:
                        o_swa[i, i_q] = torch.einsum(
                            "n h, n h v -> h v", attn_swa, v_i_swa
                        ) * g_swa_i.unsqueeze(-1)
                    else:
                        o_swa[0][cu_seqlens[i] + i_q] = torch.einsum(
                            "n h, n h v -> h v", attn_swa, v_i_swa
                        ) * g_swa_i.unsqueeze(-1)

        if head_first:
            o_slc = rearrange(o_slc, "b t h d -> b h t d")
            o_swa = rearrange(o_swa, "b t h d -> b h t d")

        return o_slc.to(dtype) + o_swa.to(dtype) if o_swa is not None else o_slc.to(dtype)


class NsaCmpFwdWorkload(WorkloadBase):
    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        group: int,
        scale: float,
        bc: int,
        bs: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        seq_lens: "list[int] | None" = None,
    ) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.group = group
        self.scale = scale
        self.bc = bc
        self.bs = bs
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.seq_lens = seq_lens

        self.head_kv = self.heads // self.group
        # chunk_num is computed during gen_inputs and stored for later use
        self.chunk_num = None

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        offsets = _packed_offsets(self.seq_lens, self.seq_num, self.c_seq_len, self.bs).to(
            torch.int32
        )

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

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        offsets: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = chunk_offsets, token_indices
        return _parallel_nsa_compression_fwd_pytorch(
            self, q, k_cmp, v_cmp, self.bs, self.scale, offsets
        )


class NsaTopkWorkload(WorkloadBase):
    def __init__(
        self,
        seq_num: int,
        c_seq_len: int,
        heads: int,
        dim: int,
        group: int,
        scale: float,
        selected_block_num: int,
        bc: int,
        bs: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        seq_lens: "list[int] | None" = None,
    ) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.group = group
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bc = bc
        self.bs = bs
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.seq_lens = seq_lens

        self.head_kv = self.heads // self.group
        # chunk_num is computed during gen_inputs and stored for later use
        self.chunk_num = None

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        offsets = _packed_offsets(self.seq_lens, self.seq_num, self.c_seq_len, 16)

        chunk_offsets = prepare_chunk_offsets(offsets, self.bs)
        token_indices = prepare_token_indices(offsets)
        chunk_num = chunk_offsets[-1].item()

        # float16, data Tie-breaking
        q = (
            torch.randn((self.c_seq_len, self.heads, self.dim), dtype=self.dtype, device="cuda")
            * 0.1
        )
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

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse: torch.Tensor,
        offsets: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> torch.Tensor:
        return _nsa_topk_torch(
            self,
            q,
            k_cmp,
            lse,
            self.selected_block_num,
            self.bs,
            self.scale,
            offsets,
            token_indices,
            chunk_offsets,
        )


class MlaDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_kv: int,
        dim: int,
        dim_pe: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device="cuda", dtype=self.dtype)
        Q_pe = torch.randn(self.batch, self.heads, self.dim_pe, device="cuda", dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device="cuda", dtype=self.dtype
        )
        K_pe = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim_pe, device="cuda", dtype=self.dtype
        )
        return Q, Q_pe, K, K_pe

    def ref_program(
        self, q: torch.Tensor, q_pe: torch.Tensor, kv: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
        - q (Tensor): [batch, heads, dim]
        - q_pe (Tensor): [batch, heads, dim_pe]
        - kv (Tensor): [batch, seqlen_kv, heads_kv, dim]
        - k_pe (Tensor): [batch, seqlen_kv, heads_kv, dim_pe]
        Outputs:
        - output (Tensor): [batch, heads, dim]
        """
        dim = q.shape[-1]
        dim_pe = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + dim_pe) ** 0.5
        Q = rearrange(
            q, "b (h g) d -> b g h d", g=num_head_groups
        )  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(
            q_pe, "b (h g) d -> b g h d", g=num_head_groups
        )  # [batch_size, num_head_groups, groups, dim_pe]

        KV = rearrange(kv, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(
            k_pe, "b n h d -> b h n d"
        )  # [batch_size, num_head_groups, groups, dim_pe]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key, "b g h d, b h s d -> b g h s"
        )  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1
        )  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(
            attention, KV, "b g h s, b h s d -> b g h d"
        )  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
        return out


class DsaDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        seq_len_kv: int,
        dim: int,
        dim_tail: int,
        topk: int,
        stride_kv: int,
        heads_kv: int,
        q_start_index_s: int,
        sm_scale: float = None,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
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
            device="cuda",
            dtype=self.dtype,
        )
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim + self.dim_tail,
            device="cuda",
            dtype=self.dtype,
        )
        indices = torch.full(
            (self.batch, self.seq_len, self.heads_kv, self.topk),
            self.seq_len_kv,
            dtype=torch.int32,
            device="cuda",
        )
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.heads_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv,
                        )
                    )[: self.topk]
                    indices[b, t, h, : len(i_i)] = i_i
        return q, kv, indices

    def selection_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the ``[batch, kv heads, q, kv]`` mask this workload's top-k selection implies.

        The mask combines the selected positions with the compressed causal limit.
        ``ref_program`` and any baseline attending over the same selection both read it,
        so the two cannot drift apart.
        """
        idx = indices.transpose(1, 2)
        b, g, sq, _ = idx.shape
        sk = self.seq_len_kv
        q_start_index_s = self.q_start_index_s
        if q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq
        device = indices.device
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=device
        ).view(-1, 1) >= torch.arange(
            self.stride_kv - 1,
            sk * self.stride_kv,
            self.stride_kv,
            dtype=torch.int32,
            device=device,
        ).view(1, -1)

        mask = torch.zeros(b, g, sq, sk + 1, dtype=torch.bool, device=device).scatter(
            3, idx.long(), 1
        )[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, : self.stride_kv - 1, 0] = True
        return mask

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        q = q.float()
        kv = kv.float()
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape

        assert kv.shape[-1] == self.dim + self.dim_tail, "you should assign dim otherwise"
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        mask = self.selection_mask(indices).view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)


def _parallel_nsa_compression_fwd_pytorch(test, q, k_cmp, v_cmp, block_size, scale, offsets):
    """PyTorch reference implementation on GPU."""
    seq_len, heads, dim_k = q.shape
    _, head_kv, _ = k_cmp.shape
    dim_v = v_cmp.shape[-1]
    group = heads // head_kv
    device = q.device
    num_seq = len(offsets) - 1

    o = torch.zeros((seq_len, heads, dim_v), dtype=torch.float32, device=device)
    lse = torch.full((seq_len, heads), float("-inf"), dtype=torch.float32, device=device)

    chunk_offsets_local = prepare_chunk_offsets(offsets, block_size)

    for i_n in range(num_seq):
        bos, eos = offsets[i_n].item(), offsets[i_n + 1].item()
        boc = chunk_offsets_local[i_n].item()

        for i_t in range(eos - bos):
            nc = (i_t + 1) // block_size
            if nc == 0:
                lse[bos + i_t] = 0.0
                continue

            q_curr = q[bos + i_t].float()
            k_curr = k_cmp[boc : boc + nc].transpose(0, 1).float()
            v_curr = v_cmp[boc : boc + nc].transpose(0, 1).float()

            k_curr = k_curr.unsqueeze(1).expand(-1, group, -1, -1).reshape(heads, nc, dim_k)
            v_curr = v_curr.unsqueeze(1).expand(-1, group, -1, -1).reshape(heads, nc, dim_v)

            scores = torch.matmul(q_curr.unsqueeze(1), k_curr.transpose(-1, -2)).squeeze(1) * scale

            m = torch.max(scores, dim=-1, keepdim=True)[0]
            exp_scores = torch.exp(scores - m)
            sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)

            probs = exp_scores / sum_exp
            out = torch.matmul(probs.unsqueeze(1), v_curr).squeeze(1)

            o[bos + i_t] = out
            lse[bos + i_t] = (m + torch.log(sum_exp)).squeeze(-1)

    return o.to(test.dtype), lse.to(test.dtype)


def _nsa_topk_torch(
    test, q, k_cmp, lse, block_counts, block_size, scale, offsets, token_indices, chunk_offsets
):
    """PyTorch reference for NSA top-k block selection."""
    _ = lse
    q = q.squeeze(0) if q.dim() == 4 else q
    k_cmp = k_cmp.squeeze(0) if k_cmp.dim() == 4 else k_cmp
    c_seq_len, heads, dim = q.shape
    head_kv = k_cmp.shape[1]
    group = heads // head_kv
    selected_block_num = (
        block_counts if isinstance(block_counts, int) else block_counts.max().item()
    )
    bs = block_size
    LOG2_E = 1.44269504
    scale_log2 = scale * LOG2_E

    device = q.device
    accum_dtype = torch.float32

    lse_out = torch.zeros((c_seq_len, heads), dtype=accum_dtype, device=device)
    block_indices = torch.zeros(
        (c_seq_len, head_kv, selected_block_num), dtype=torch.int32, device=device
    )

    for i_c in range(c_seq_len):
        i_n, i_t = token_indices[i_c, 0].item(), token_indices[i_c, 1].item()
        bos = offsets[i_n].item()
        boc = chunk_offsets[i_n].item()
        nc = (i_t + 1) // bs
        q_curr = q[bos + i_t]

        for i_h in range(head_kv):
            q_h = q_curr[i_h * group : (i_h + 1) * group]
            scores_max = torch.full((group,), float("-inf"), dtype=accum_dtype, device=device)
            logsum = torch.zeros((group,), dtype=accum_dtype, device=device)

            for i_loop in range(0, nc, bs):
                start_idx = i_loop
                end_idx = min(start_idx + bs, nc)
                curr_bc = end_idx - start_idx
                k_blocks = k_cmp[boc + start_idx : boc + end_idx, i_h]
                acc_s = torch.matmul(q_h, k_blocks.t()).to(accum_dtype)
                if curr_bc < bs:
                    padding = torch.full(
                        (group, bs - curr_bc), float("-inf"), dtype=accum_dtype, device=device
                    )
                    acc_s = torch.cat([acc_s, padding], dim=1)
                o_c = torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device)
                valid_mask = o_c < nc
                acc_s = torch.where(
                    valid_mask.unsqueeze(0), acc_s, torch.full_like(acc_s, float("-inf"))
                )
                scores_max_prev = scores_max.clone()
                scores_max_curr = acc_s.max(dim=1)[0]
                scores_max = torch.maximum(scores_max, scores_max_curr)
                scores_scale = torch.exp2((scores_max_prev - scores_max) * scale_log2)
                acc_s_exp = torch.exp2((acc_s - scores_max.unsqueeze(1)) * scale_log2)
                acc_s_exp = torch.where(
                    acc_s > float("-inf"), acc_s_exp, torch.zeros_like(acc_s_exp)
                )
                logsum = logsum * scores_scale + acc_s_exp.sum(dim=1)

            if nc == 0:
                b_lse = torch.zeros((group,), dtype=accum_dtype, device=device)
            else:
                logsum_log2 = torch.where(
                    logsum > 0,
                    torch.log2(logsum),
                    torch.full((group,), float("-inf"), dtype=accum_dtype, device=device),
                )
                b_lse = (scores_max * scale_log2 + logsum_log2) / LOG2_E
                b_lse = torch.where(logsum <= 0, torch.zeros_like(b_lse), b_lse)
            lse_out[bos + i_t, i_h * group : (i_h + 1) * group] = b_lse

            nc_topk = i_t // bs + 1
            pool_scores = torch.full((bs * 2,), float("-inf"), dtype=accum_dtype, device=device)
            pool_indices = torch.zeros((bs * 2,), dtype=torch.int32, device=device)

            for i_tk in range(0, nc_topk, bs):
                start_idx = i_tk
                end_idx = min(start_idx + bs, nc_topk)
                curr_bc_tk = end_idx - start_idx
                k_blocks = k_cmp[boc + start_idx : boc + end_idx, i_h]
                acc_s = torch.matmul(q_h, k_blocks.t()).to(accum_dtype)
                if curr_bc_tk < bs:
                    padding = torch.full(
                        (group, bs - curr_bc_tk), float("-inf"), dtype=accum_dtype, device=device
                    )
                    acc_s = torch.cat([acc_s, padding], dim=1)
                o_c = torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device)
                is_curr = o_c == i_t // bs
                is_hist = o_c < i_t // bs
                importance = torch.where(
                    is_curr.unsqueeze(0),
                    torch.ones((group, bs), dtype=accum_dtype, device=device),
                    torch.where(
                        is_hist.unsqueeze(0),
                        torch.exp2((acc_s * scale - b_lse.unsqueeze(1)) * LOG2_E),
                        torch.zeros((group, bs), dtype=accum_dtype, device=device),
                    ),
                )
                b_i_current = importance.sum(dim=0)
                pool_scores[bs : bs + bs] = b_i_current
                pool_indices[bs : bs + bs] = (
                    torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device) + 1
                )
                o_c_valid = (
                    torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device)
                    < nc_topk
                )
                pool_scores[bs : bs + bs] = torch.where(
                    o_c_valid,
                    pool_scores[bs : bs + bs],
                    torch.full_like(pool_scores[bs : bs + bs], float("-inf")),
                )
                pool_indices[bs : bs + bs] = torch.where(
                    o_c_valid,
                    pool_indices[bs : bs + bs],
                    torch.zeros_like(pool_indices[bs : bs + bs]),
                )
                eps_val, score_scale = 1e-5, 1e12
                scores_quantized = (pool_scores / eps_val).round() * eps_val
                sort_key = scores_quantized.to(torch.float64) * score_scale + pool_indices.to(
                    torch.float64
                )
                sort_key = torch.where(
                    pool_indices > 0,
                    sort_key,
                    torch.full_like(sort_key, float("-inf"), dtype=torch.float64),
                )
                sorted_indices = torch.argsort(sort_key, descending=True)
                pool_scores = pool_scores[sorted_indices]
                pool_indices = pool_indices[sorted_indices]

            final_indices = pool_indices[:selected_block_num] - 1
            final_indices = torch.where(
                final_indices >= 0,
                final_indices,
                torch.tensor(-1, dtype=torch.int32, device=device),
            )
            block_indices[i_c, i_h, :selected_block_num] = final_indices.to(torch.int32)

    return block_indices
