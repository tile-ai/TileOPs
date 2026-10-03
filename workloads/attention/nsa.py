import torch
from einops import einsum, repeat

from workloads.device import run_device
from workloads.sequence_metadata import prepare_chunk_offsets, prepare_token_indices
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "NsaCmpFwdCall",
    "NsaCmpFwdWorkload",
    "NsaFwdCall",
    "NsaFwdWorkload",
    "NsaTopkCall",
    "NsaTopkWorkload",
]


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
        return bounds.to(run_device())
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
        .to(run_device())
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
        self.seq_lens = seq_lens

        self.head_kv = self.heads // self.groups

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        # block_counts and block_indices decide how much this call reads, so its
        # roofline moves with them. They come from this workload's own generator,
        # not the global stream, which a draw added anywhere upstream shifts.
        g = self.rng(device=run_device())
        offsets = _packed_offsets(self.seq_lens, self.batch, self.c_seq_len, 16)

        def _shuffled(n_head: int) -> torch.Tensor:
            perm = torch.randperm(self.c_seq_len, device=run_device(), generator=g)
            return (
                torch.linspace(0, 1, steps=self.c_seq_len, dtype=self.dtype, device=run_device())[
                    perm
                ]
                .view(self.c_seq_len, 1, 1)
                .expand(self.c_seq_len, n_head, self.dim)
                .clone()
                .requires_grad_(True)
            )

        q, k, v = _shuffled(self.heads), _shuffled(self.head_kv), _shuffled(self.head_kv)
        self.g_slc = torch.ones(
            (self.batch, self.c_seq_len, self.heads), dtype=self.dtype, device=run_device()
        ).requires_grad_(True)

        token_indices = prepare_token_indices(offsets)
        # How many blocks each token may attend to: causally the one it sits in and those
        # before, otherwise every block of its sequence.
        if self.is_causal:
            chunks = ((token_indices[:, 1] + self.block_size - 1) // self.block_size).clamp(min=1)
        else:
            seq_len = (offsets[1:] - offsets[:-1])[token_indices[:, 0]]
            chunks = (seq_len + self.block_size - 1) // self.block_size
        n_cand = max(int(chunks.max().item()), self.selected_blocks)
        # Each token picks selected_blocks distinct blocks out of its candidates. Sorting
        # one random key per candidate is a batched randperm; the ineligible tail sorts
        # last under +inf, and a pick that reaches there becomes a c_seq_len slot, which
        # the kernel and the reference both read as "attends to nothing".
        keys = torch.rand(
            (self.c_seq_len, self.head_kv, n_cand), device=run_device(), generator=g
        ).masked_fill(
            torch.arange(n_cand, device=run_device()) >= chunks[:, None, None], float("inf")
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
            device=run_device(),
            generator=g,
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

            # A causal token sees keys up to itself, a non-causal one up to the sequence end.
            i_q = torch.arange(n_token, device=device).view(n_token, 1, 1)
            beyond = pos > i_q if self.is_causal else pos >= n_token
            attn = (
                einsum(q_b * scale, k_slc, "t h d, t n h d -> t n h")
                .masked_fill(
                    (pos < 0) | beyond | (slot_block >= block_counts[bos:eos].unsqueeze(1)),
                    float("-inf"),
                )
                .softmax(1)
                .nan_to_num(0.0)  # a token no selected block gives a key outputs zeros
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
        bs: int,
        dtype: torch.dtype,
        seq_lens: "list[int] | None" = None,
    ) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.group = group
        self.scale = scale
        self.bs = bs
        self.dtype = dtype
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
        q = torch.randn(
            (self.c_seq_len, self.heads, self.dim_k), dtype=self.dtype, device=run_device()
        )
        k = torch.randn(
            (chunk_num, self.head_kv, self.dim_k), dtype=self.dtype, device=run_device()
        )
        v = torch.randn(
            (chunk_num, self.head_kv, self.dim_v), dtype=self.dtype, device=run_device()
        )

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
        bs: int,
        dtype: torch.dtype,
        seq_lens: "list[int] | None" = None,
    ) -> None:
        self.seq_num = seq_num
        self.c_seq_len = c_seq_len
        self.heads = heads
        self.dim = dim
        self.group = group
        self.scale = scale
        self.selected_block_num = selected_block_num
        self.bs = bs
        self.dtype = dtype
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
            torch.randn(
                (self.c_seq_len, self.heads, self.dim), dtype=self.dtype, device=run_device()
            )
            * 0.1
        )
        k = (
            torch.randn((chunk_num, self.head_kv, self.dim), dtype=self.dtype, device=run_device())
            * 0.1
        )

        q.requires_grad_(True)
        k.requires_grad_(True)

        self.chunk_num = chunk_offsets[-1].item()
        return (
            q,
            k,
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        offsets: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> torch.Tensor:
        return _nsa_topk_torch(
            self,
            q,
            k_cmp,
            self.selected_block_num,
            self.bs,
            self.scale,
            offsets,
            token_indices,
            chunk_offsets,
        )


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
    test, q, k_cmp, block_counts, block_size, scale, offsets, token_indices, chunk_offsets
):
    """PyTorch reference for NSA top-k block selection."""
    _ = token_indices
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


class NsaCmpFwdCall(CallWorkload, NsaCmpFwdWorkload):
    """A manifest call of NSACompressedVarlenFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NsaCmpFwdWorkload.__init__(
            self,
            ix["N"],
            ix["T_q"],
            ix["H"],
            ix["DK"],
            ix["DV"],
            ix["H"] // ix["H_kv"],
            params["scale"],
            params["bs"],
            getattr(torch, ix["T"]),
        )
        self.chunk_num = ix["C"]

    gen_inputs = CallWorkload.gen_inputs


class NsaTopkCall(CallWorkload, NsaTopkWorkload):
    """A manifest call of NSATopKVarlenFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NsaTopkWorkload.__init__(
            self,
            ix["N"],
            ix["T_q"],
            ix["H"],
            ix["D"],
            ix["H"] // ix["H_kv"],
            params["scale"],
            params["selected_block_num"],
            params["bs"],
            getattr(torch, ix["T"]),
        )
        self.chunk_num = ix["C"]

    gen_inputs = CallWorkload.gen_inputs


class NsaFwdCall(CallWorkload, NsaFwdWorkload):
    """A manifest call of NSAVarlenFwdOp; the row's generators make the selection."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NsaFwdWorkload.__init__(
            self,
            ix["N"],
            ix["H"],
            ix["T_q"],
            ix["D"],
            params["is_causal"],
            params["scale"],
            params["block_size"],
            ix["H"] // ix["H_kv"],
            ix["SEL"],
            getattr(torch, ix["T"]),
        )
        # The reference scales the selected branch by its gate, which this op does not take.
        self.g_slc = torch.ones(
            (ix["N"], ix["T_q"], ix["H"]), dtype=self.dtype, device=run_device()
        )

    gen_inputs = CallWorkload.gen_inputs
