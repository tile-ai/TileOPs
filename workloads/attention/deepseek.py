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

        def _shuffled(n_head: int) -> torch.Tensor:
            perm = torch.randperm(self.c_seq_len, device="cuda")
            return (
                torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype, device="cuda")[perm]
                .view(self.c_seq_len, 1, 1)
                .expand(self.c_seq_len, n_head, self.dim)
                .clone()
                .requires_grad_(True)
            )

        q, k, v = _shuffled(self.heads), _shuffled(self.head_kv), _shuffled(self.head_kv)
        self.g_slc = torch.ones(
            (self.batch, self.c_seq_len, self.heads), dtype=self.dtype, device="cuda"
        ).requires_grad_(True)

        token_indices = prepare_token_indices(offsets)
        # How many blocks each token may attend to: the one it sits in, and those before.
        chunks = ((token_indices[:, 1] + self.block_size - 1) // self.block_size).clamp(min=1)
        n_cand = max(int(chunks.max().item()), self.selected_blocks)
        # Each token picks selected_blocks distinct blocks out of its candidates. Sorting
        # one random key per candidate is a batched randperm; the ineligible tail sorts
        # last under +inf, and a pick that reaches there becomes a c_seq_len slot, which
        # the kernel and the reference both read as "attends to nothing".
        keys = torch.rand((self.c_seq_len, self.head_kv, n_cand), device="cuda").masked_fill(
            torch.arange(n_cand, device="cuda") >= chunks[:, None, None], float("inf")
        )
        picked = keys.argsort(-1)[..., : self.selected_blocks]
        block_indices = (
            torch.where(picked < chunks[:, None, None], picked, self.c_seq_len)
            .to(torch.int32)
            .sort(-1)[0]
        )
        block_counts = torch.randint(
            1,
            self.selected_blocks + 1,
            (self.c_seq_len, self.head_kv),
            dtype=torch.int32,
            device="cuda",
        )
        return (
            q,
            k,
            v,
            block_indices,
            block_counts,
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
        dtype, device = q.dtype, q.device
        bs = self.block_size
        scale = self.scale if self.scale is not None else k.shape[-1] ** -0.5
        group = q.shape[1] // k.shape[1]
        n_slot = block_indices.shape[-1] * bs

        k, v, block_indices = (
            repeat(x, "t h d -> t (h g) d", g=group) for x in (k, v, block_indices)
        )
        block_counts = repeat(block_counts, "t h -> t (h g)", g=group)
        q, k, v = (x.float() for x in (q, k, v))
        heads = q.shape[1]
        # Which selected block each gathered position came from, for the count mask.
        slot_block = (torch.arange(n_slot, device=device) // bs).view(1, n_slot, 1)
        head = torch.arange(heads, device=device)

        o_slc = torch.zeros_like(v)
        for i in range(len(offsets) - 1):
            bos, eos = offsets[i].item(), offsets[i + 1].item()
            n_token = eos - bos
            q_b, k_b, v_b = q[bos:eos], k[bos:eos], v[bos:eos]

            # [t, s*bs, hq]: the token each selected block contributes, per head.
            pos = block_indices[bos:eos].unsqueeze(-1) * bs + torch.arange(bs, device=device)
            pos = pos.view(n_token, heads, n_slot).transpose(1, 2)
            # Out-of-range positions are masked below; clamping only keeps the gather legal.
            k_slc, v_slc = (x[pos.clamp(0, n_token - 1), head] for x in (k_b, v_b))

            i_q = torch.arange(n_token, device=device).view(n_token, 1, 1)
            attn = (
                einsum(q_b * scale, k_slc, "t h d, t n h d -> t n h")
                .masked_fill(
                    (pos < 0) | (pos > i_q) | (slot_block >= block_counts[bos:eos].unsqueeze(1)),
                    float("-inf"),
                )
                .softmax(1)
            )
            o_slc[bos:eos] = einsum(attn, v_slc, "t n h, t n h v -> t h v") * self.g_slc[
                0, bos:eos
            ].unsqueeze(-1)

        return o_slc.to(dtype)


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

    # A token before the first block closed attends to nothing; both outputs stay zero.
    o = torch.zeros((seq_len, heads, dim_v), dtype=torch.float32, device=device)
    lse = torch.zeros((seq_len, heads), dtype=torch.float32, device=device)

    chunk_offsets_local = prepare_chunk_offsets(offsets, block_size)

    for i_n in range(num_seq):
        bos, eos = offsets[i_n].item(), offsets[i_n + 1].item()
        boc = chunk_offsets_local[i_n].item()
        n_token = eos - bos
        # Blocks the last token attends to; every earlier token attends to a prefix.
        n_chunk = n_token // block_size
        if n_chunk == 0:
            continue

        nc = (torch.arange(n_token, device=device) + 1) // block_size
        q_seq = q[bos:eos].float().view(n_token, head_kv, group, dim_k)
        k_seq = k_cmp[boc : boc + n_chunk].float()
        v_seq = v_cmp[boc : boc + n_chunk].float()

        scores = einsum(q_seq, k_seq, "t h g d, n h d -> t h g n") * scale
        scores = scores.masked_fill(
            torch.arange(n_chunk, device=device) >= nc[:, None, None, None], float("-inf")
        )
        m = scores.max(dim=-1, keepdim=True)[0]
        reached = scores > float("-inf")
        exp_scores = torch.where(reached, torch.exp(torch.where(reached, scores - m, 0.0)), 0.0)
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)
        out = einsum(exp_scores / sum_exp, v_seq, "t h g n, n h v -> t h g v")

        # A token whose blocks have not closed divides 0 by 0 above; where() writes the
        # zeros the kernel writes rather than that quotient.
        attends = (nc > 0).view(n_token, 1, 1)
        o[bos:eos] = torch.where(attends.unsqueeze(-1), out, 0.0).reshape(n_token, heads, dim_v)
        lse[bos:eos] = torch.where(attends, (m + torch.log(sum_exp)).squeeze(-1), 0.0).reshape(
            n_token, heads
        )

    return o.to(test.dtype), lse.to(test.dtype)


def _nsa_topk_torch(
    test, q, k_cmp, lse, block_counts, block_size, scale, offsets, token_indices, chunk_offsets
):
    """PyTorch reference for NSA top-k block selection."""
    _ = lse, token_indices
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

    # The kernel ranks bs candidates at a time against a pool that retains the best bs
    # seen so far, so its first bs slots are the ranking over every candidate and the
    # rest are whichever batch it read last. Only the first bs are a defined answer.
    if selected_block_num > bs:
        raise ValueError("selected_block_num must be no larger than block_size")

    # A slot no candidate block reaches reads as -1.
    block_indices = torch.full(
        (c_seq_len, head_kv, selected_block_num), -1, dtype=torch.int32, device=device
    )

    for i_n in range(len(offsets) - 1):
        bos, eos = offsets[i_n].item(), offsets[i_n + 1].item()
        boc = chunk_offsets[i_n].item()
        n_token = eos - bos
        # Blocks the last token ranks; every earlier token ranks a prefix of them.
        n_chunk = (n_token - 1) // bs + 1

        i_t = torch.arange(n_token, device=device)
        nc = ((i_t + 1) // bs)[:, None, None, None]  # blocks closed before the token
        curr = (i_t // bs)[:, None, None, None]  # the block the token sits in
        o_c = torch.arange(n_chunk, device=device)

        q_seq = q[bos:eos].view(n_token, head_kv, group, dim)
        k_seq = k_cmp[boc : boc + n_chunk]
        acc_s = einsum(q_seq, k_seq, "t h g d, n h d -> t h g n").to(accum_dtype)

        # The log-sum-exp over the closed blocks, which the kernel's running softmax
        # arrives at the same way. A token with none of them attends to nothing, and
        # taking exp2 only where a block is attended keeps that row out of NaN.
        attended = acc_s.masked_fill(o_c >= nc, float("-inf"))
        scores_max = attended.max(dim=-1, keepdim=True)[0]
        reached = attended > float("-inf")
        shifted = torch.where(reached, (attended - scores_max) * scale_log2, 0.0)
        logsum = torch.where(reached, torch.exp2(shifted), 0.0).sum(dim=-1, keepdim=True)
        b_lse = torch.where(nc > 0, (scores_max * scale_log2 + torch.log2(logsum)) / LOG2_E, 0.0)

        # A closed block ranks by the share of the token's attention it holds. The block
        # the token sits in scores group, which no closed block can reach.
        importance = torch.where(
            o_c == curr,
            1.0,
            torch.where(o_c < curr, torch.exp2((acc_s * scale - b_lse) * LOG2_E), 0.0),
        ).sum(dim=2)

        # Quantizing the score and adding the block ordinal makes the ranking total, so
        # equal scores break the same way here and in the kernel.
        eps, score_scale = 1e-5, 1e12
        sort_key = (importance / eps).round().to(torch.float64) * eps * score_scale + o_c
        sort_key = sort_key.masked_fill(o_c > curr.squeeze(-1), float("-inf"))

        n_pick = min(selected_block_num, n_chunk)
        top = sort_key.topk(n_pick, dim=-1)
        block_indices[bos:eos, :, :n_pick] = torch.where(
            top.values > float("-inf"), top.indices.to(torch.int32), -1
        )

    return block_indices
