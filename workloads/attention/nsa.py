import torch
from einops import einsum, repeat

from workloads.device import run_device
from workloads.sequence_metadata import prepare_chunk_offsets, prepare_token_indices
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "NSACompressedFwdCall",
    "NSACompressedFwdWorkload",
    "NSAFwdCall",
    "NSAFwdWorkload",
    "NSATopKCall",
    "NSATopKWorkload",
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


class NSAFwdWorkload(WorkloadBase):
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

    def verification(self, *inputs):
        from workloads.numerics import Exact

        # Both tensor-core implementations narrow unnormalized softmax weights
        # before P @ V, then narrow the output again. Cancellation therefore
        # needs the storage-dtype absolute bound as well as its relative bound.
        return Exact()


class NSACompressedFwdWorkload(WorkloadBase):
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

    def verification(self, *inputs):
        from workloads.numerics import Exact, reference_tolerance, zeroed_input

        tol = (
            reference_tolerance(inputs[0].dtype)
            if inputs[0].dtype == torch.bfloat16
            else {"atol": 4e-3, "rtol": 1e-5}
        )
        return Exact(controls=(zeroed_input(0, "first-input-zeroed"),), **tol)


class NSATopKWorkload(WorkloadBase):
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
        inputs = q, k_cmp, offsets, chunk_offsets, token_indices
        result = torch.full(
            (q.shape[0], k_cmp.shape[1], self.selected_block_num),
            -1,
            dtype=torch.int32,
            device=q.device,
        )
        for tokens, scores in self._scores(*inputs):
            top = scores.topk(min(self.selected_block_num, scores.shape[-1]), dim=-1)
            result[tokens, :, : top.indices.shape[-1]] = torch.where(
                top.values > -float("inf"), top.indices.to(torch.int32), -1
            )
        return result

    def _scores(self, q, k_cmp, offsets, chunk_offsets, _token_indices):
        for i in range(len(offsets) - 1):
            bos, eos = offsets[i].item(), offsets[i + 1].item()
            boc = chunk_offsets[i].item()
            n_chunk = (eos - bos + self.bs - 1) // self.bs
            yield (
                slice(bos, eos),
                _nsa_topk_scores(q[bos:eos], k_cmp[boc : boc + n_chunk], self.bs, self.scale),
            )

    def selection_scores(self, indices: torch.Tensor, *inputs) -> torch.Tensor:
        """Reference importance at each selected block, with -inf for padding."""
        selected = torch.empty_like(indices, dtype=torch.float32)
        for tokens, scores in self._scores(*inputs):
            picked = indices[tokens].long()
            selected[tokens] = scores.gather(-1, picked.clamp_min(0)).masked_fill(
                picked < 0, -float("inf")
            )
        return selected

    def verification(self, *inputs):
        from workloads.numerics import Custom, NegativeControl

        def validate(got, expected):
            assert torch.equal(got == -1, expected == -1), "unfilled top-k slots differ"
            current = inputs[-1][:, 1, None, None] // self.bs
            assert ((got >= -1) & (got <= current)).all(), "non-causal or invalid block id"
            ordered = got.sort(-1).values
            assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any(), (
                "duplicate selected block"
            )
            torch.testing.assert_close(
                self.selection_scores(got, *inputs),
                self.selection_scores(expected, *inputs),
                rtol=1e-5,
                atol=1e-6,
            )

        return Custom(
            validate,
            "valid top-k indices and selected scores",
            # Selecting every visible block is independent of Q, and tied scores
            # can keep their order after zeroing Q. An invalid index is a fault
            # for every nonempty selection, including those valid corner cases.
            controls=(
                NegativeControl(
                    "invalid-block-index", lambda ref, args: torch.full_like(ref(*args), -2)
                ),
            ),
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


def _nsa_topk_scores(q, k_cmp, block_size, scale):
    """Independent FP32 importance scores for one sequence, with FLA's Q scaling."""
    n_token, heads, dim = q.shape
    n_chunk, head_kv, _ = k_cmp.shape
    group = heads // head_kv
    q_scaled = (q * scale).float().view(n_token, head_kv, group, dim)
    scores = einsum(q_scaled, k_cmp.float(), "t h g d, n h d -> t h g n")
    position = torch.arange(n_token, device=q.device)[:, None, None, None]
    block = torch.arange(n_chunk, device=q.device)
    closed = (position + 1) // block_size
    current = position // block_size
    lse = scores.masked_fill(block >= closed, -float("inf")).logsumexp(-1, keepdim=True)
    lse = torch.where(closed > 0, lse, 0.0)
    priority = (block == 0) | (block == current - 1) | (block == current)
    probability = (scores.masked_fill(block >= current, -float("inf")) - lse).exp()
    importance = torch.where(priority, 1.0, probability).sum(dim=2)
    return importance.masked_fill(block > current.squeeze(-1), -float("inf"))


class NSACompressedFwdCall(CallWorkload, NSACompressedFwdWorkload):
    """A manifest call of NSACompressedVarlenFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NSACompressedFwdWorkload.__init__(
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


class NSATopKCall(CallWorkload, NSATopKWorkload):
    """A manifest call of NSATopKVarlenFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NSATopKWorkload.__init__(
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


class NSAFwdCall(CallWorkload, NSAFwdWorkload):
    """A manifest call of NSAVarlenFwdOp; the row's generators make the selection."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        NSAFwdWorkload.__init__(
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
