import torch
from einops import einsum

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
        kv_lens: "list[int] | None" = None,
        varied_dims: bool = False,
        selection: str = "random",
    ) -> None:
        """``kv_lens`` gives each request its own keys, of which its ``seq_lens`` queries
        are the last, passed as ``cu_seqlens_k``; ``None`` makes the keys the queries'
        tokens. ``varied_dims`` draws q, k and v at random along the head dimension too,
        where the default ramps vary only across tokens.

        ``selection`` is ``"random"``: distinct visible blocks per KV head, sorted, and a
        random count; ``"topk"``: what ``TopKSelectFwdOp`` returns for random logits over
        the blocks before the token's own, in no order and padded with the request's
        block count, then the token's own block, one list for every KV head and every
        slot counted; or ``"padding"``: every slot holds the padding id, so no token sees
        a key."""
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
        self.kv_lens = kv_lens
        self.varied_dims = varied_dims
        self.selection = selection

        self.head_kv = self.heads // self.groups

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        # block_counts and block_indices decide how much this call reads, so its
        # roofline moves with them. They come from this workload's own generator,
        # not the global stream, which a draw added anywhere upstream shifts.
        g = self.rng(device=run_device())
        cu_seqlens = _packed_offsets(self.seq_lens, self.batch, self.c_seq_len, 16)
        if self.kv_lens is None:
            cu_seqlens_k, n_keys = cu_seqlens, self.c_seq_len
        else:
            cu_seqlens_k = _packed_offsets(self.kv_lens, self.batch, sum(self.kv_lens), 16)
            n_keys = sum(self.kv_lens)

        def _shuffled(rows: int, n_head: int) -> torch.Tensor:
            if self.varied_dims:
                return torch.randn((rows, n_head, self.dim), device=run_device(), generator=g).to(
                    self.dtype
                )
            perm = torch.randperm(rows, device=run_device(), generator=g)
            return (
                torch.linspace(0, 1, steps=rows, dtype=self.dtype, device=run_device())[perm]
                .view(rows, 1, 1)
                .expand(rows, n_head, self.dim)
                .clone()
                .requires_grad_(True)
            )

        q = _shuffled(self.c_seq_len, self.heads)
        k, v = _shuffled(n_keys, self.head_kv), _shuffled(n_keys, self.head_kv)
        self.g_slc = torch.ones(
            (self.batch, self.c_seq_len, self.heads), dtype=self.dtype, device=run_device()
        ).requires_grad_(True)

        token_indices = prepare_token_indices(cu_seqlens)
        # A request's queries are the last of its keys.
        n_key = (cu_seqlens_k[1:] - cu_seqlens_k[:-1])[token_indices[:, 0]]
        n_query = (cu_seqlens[1:] - cu_seqlens[:-1])[token_indices[:, 0]]
        position = token_indices[:, 1] + n_key - n_query
        # How many blocks each token may attend to: causally the one it sits in and those
        # before, otherwise every block of its sequence.
        if self.is_causal:
            chunks = ((position + self.block_size - 1) // self.block_size).clamp(min=1)
        else:
            chunks = (n_key + self.block_size - 1) // self.block_size
        n_cand = max(int(chunks.max().item()), self.selected_blocks)
        # Each token picks selected_blocks distinct blocks out of its candidates. Sorting
        # one random key per candidate is a batched randperm; the ineligible tail sorts
        # last under +inf, and a pick that reaches there becomes a slot past every key,
        # which the kernel and the reference both read as "attends to nothing".
        keys = torch.rand(
            (self.c_seq_len, self.head_kv, n_cand), device=run_device(), generator=g
        ).masked_fill(
            torch.arange(n_cand, device=run_device()) >= chunks[:, None, None], float("inf")
        )
        picked = keys.argsort(-1)[..., : self.selected_blocks]
        block_indices = (
            torch.where(picked < chunks[:, None, None], picked, n_keys).to(torch.int32).sort(-1)[0]
        )
        block_counts = torch.randint(
            1,
            self.selected_blocks + 1,
            (self.c_seq_len, self.head_kv),
            dtype=torch.int32,
            device=run_device(),
            generator=g,
        )
        shape = (self.c_seq_len, self.head_kv, self.selected_blocks)
        if self.selection == "padding":
            block_indices = torch.full(shape, n_keys, dtype=torch.int32, device=run_device())
        elif self.selection == "topk":
            own = position // self.block_size
            blocks = (n_key + self.block_size - 1) // self.block_size
            top_k = self.selected_blocks - 1
            n_cand = int(blocks.max().item())
            logits = torch.rand((self.c_seq_len, n_cand), device=run_device(), generator=g)
            before = torch.arange(n_cand, device=run_device()) < own[:, None]
            score, picked = logits.masked_fill(~before, -1.0).topk(min(top_k, n_cand), dim=-1)
            picked = torch.where(score >= 0, picked, blocks[:, None])
            if picked.shape[-1] < top_k:
                fill = blocks[:, None].expand(-1, top_k - picked.shape[-1])
                picked = torch.cat([picked, fill], -1)
            # No order along the list: a random permutation per token.
            order = torch.rand(picked.shape, device=run_device(), generator=g).argsort(-1)
            listed = torch.cat([picked.gather(-1, order), own[:, None]], -1).to(torch.int32)
            block_indices = listed[:, None, :].expand(shape).contiguous()
        if self.selection != "random":
            # Every slot counts: a padding id alone must select nothing.
            block_counts = torch.full(
                shape[:2], self.selected_blocks, dtype=torch.int32, device=run_device()
            )
        inputs = (
            q,
            k,
            v,
            block_indices,
            block_counts,
            cu_seqlens.to(torch.int32),
            token_indices.to(torch.int32),
        )
        if self.kv_lens is None:
            return inputs
        return (*inputs, cu_seqlens_k.to(torch.int32))

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_indices: torch.Tensor,
        block_counts: torch.Tensor,
        cu_seqlens: torch.Tensor,
        token_indices: torch.Tensor,
        cu_seqlens_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = token_indices
        cu_seqlens_k = cu_seqlens if cu_seqlens_k is None else cu_seqlens_k
        dtype, device = q.dtype, q.device
        bs = self.block_size
        scale = self.scale if self.scale is not None else k.shape[-1] ** -0.5
        heads_kv, dim = k.shape[1], k.shape[2]
        group = q.shape[1] // heads_kv
        n_slot = block_indices.shape[-1] * bs
        q, k, v = (x.float() for x in (q, k, v))
        # Which selected block each gathered position came from, for the count mask.
        slot_block = (torch.arange(n_slot, device=device) // bs).view(1, 1, n_slot)
        head = torch.arange(heads_kv, device=device).view(1, heads_kv, 1)
        # Tokens a step gathers, bounding its k and v rows to 256 MiB.
        step = max(1, (256 * 1024**2) // (2 * heads_kv * max(n_slot, 1) * dim * 4))

        o_slc = q.new_zeros(q.shape[0], q.shape[1], v.shape[-1])
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            bos_k, eos_k = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
            n_token, n_key = eos - bos, eos_k - bos_k
            k_b, v_b = k[bos_k:eos_k], v[bos_k:eos_k]
            for lo in range(bos, eos, step):
                hi = min(lo + step, eos)
                # [t, h_kv, s*bs]: the key each selected block contributes, per KV head.
                # int64: a padding id times the block size may pass the int32 range.
                pos = block_indices[lo:hi].long().unsqueeze(-1) * bs + torch.arange(
                    bs, device=device
                )
                pos = pos.flatten(2)
                # Out-of-range positions are masked below; clamping keeps the gather legal.
                k_slc, v_slc = (x[pos.clamp(0, n_key - 1), head] for x in (k_b, v_b))
                # A causal token sees keys up to itself, a non-causal one up to the sequence
                # end; the queries are the last n_token of the n_key keys.
                i_q = torch.arange(lo - bos, hi - bos, device=device) + n_key - n_token
                beyond = pos > i_q.view(-1, 1, 1) if self.is_causal else pos >= n_key
                dead = (pos < 0) | beyond | (slot_block >= block_counts[lo:hi].unsqueeze(-1))
                q_g = q[lo:hi].view(hi - lo, heads_kv, group, dim) * scale
                attn = (
                    einsum(q_g, k_slc, "t k g d, t k n d -> t k g n")
                    .masked_fill(dead.unsqueeze(2), float("-inf"))
                    .softmax(-1)
                    .nan_to_num(0.0)  # a token no selected block gives a key outputs zeros
                )
                out = einsum(attn, v_slc, "t k g n, t k n v -> t k g v").flatten(1, 2)
                o_slc[lo:hi] = out * self.g_slc[0, lo:hi].unsqueeze(-1)

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
        cu_seqlens = _packed_offsets(self.seq_lens, self.seq_num, self.c_seq_len, self.bs).to(
            torch.int32
        )

        chunk_offsets = prepare_chunk_offsets(cu_seqlens, self.bs).to(torch.int32)
        token_indices = prepare_token_indices(cu_seqlens).to(torch.int32)
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
            cu_seqlens.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        v_cmp: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = chunk_offsets, token_indices
        return _parallel_nsa_compression_fwd_pytorch(
            self, q, k_cmp, v_cmp, self.bs, self.scale, cu_seqlens
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
        cu_seqlens = _packed_offsets(self.seq_lens, self.seq_num, self.c_seq_len, 16)

        chunk_offsets = prepare_chunk_offsets(cu_seqlens, self.bs)
        token_indices = prepare_token_indices(cu_seqlens)
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
            cu_seqlens.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        cu_seqlens: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> torch.Tensor:
        inputs = q, k_cmp, cu_seqlens, chunk_offsets, token_indices
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

    def _scores(self, q, k_cmp, cu_seqlens, chunk_offsets, _token_indices):
        for i in range(len(cu_seqlens) - 1):
            bos, eos = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
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


def _parallel_nsa_compression_fwd_pytorch(test, q, k_cmp, v_cmp, block_size, scale, cu_seqlens):
    """PyTorch reference implementation on GPU."""
    seq_len, heads, dim_k = q.shape
    _, head_kv, _ = k_cmp.shape
    dim_v = v_cmp.shape[-1]
    group = heads // head_kv
    device = q.device
    scale = dim_k**-0.5 if scale is None else scale
    num_seq = len(cu_seqlens) - 1

    # A token before the first block closed attends to nothing: o zero, lse -inf.
    o = torch.zeros((seq_len, heads, dim_v), dtype=torch.float32, device=device)
    lse = torch.full((seq_len, heads), float("-inf"), dtype=torch.float32, device=device)

    chunk_offsets_local = prepare_chunk_offsets(cu_seqlens, block_size)

    for i_n in range(num_seq):
        bos, eos = cu_seqlens[i_n].item(), cu_seqlens[i_n + 1].item()
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
        # zero output and -inf lse the kernel writes rather than that quotient.
        attends = (nc > 0).view(n_token, 1, 1)
        o[bos:eos] = torch.where(attends.unsqueeze(-1), out, 0.0).reshape(n_token, heads, dim_v)
        lse[bos:eos] = torch.where(
            attends, (m + torch.log(sum_exp)).squeeze(-1), float("-inf")
        ).reshape(n_token, heads)

    # The lse is float32 and head-major, as the op returns it.
    return o.to(test.dtype), lse.T.contiguous()


def _nsa_topk_scores(q, k_cmp, block_size, scale):
    """Independent FP32 importance scores for one sequence, with FLA's Q scaling."""
    n_token, heads, dim = q.shape
    n_chunk, head_kv, _ = k_cmp.shape
    group = heads // head_kv
    scale = dim**-0.5 if scale is None else scale
    q_scaled = (q.float() * scale).to(q.dtype).float().view(n_token, head_kv, group, dim)
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
        indices, params = call.indices, call.params
        NSACompressedFwdWorkload.__init__(
            self,
            indices["N"],
            indices["T_q"],
            indices["H"],
            indices["DK"],
            indices["DV"],
            indices["H"] // indices["H_kv"],
            params["sm_scale"],
            params["block_size"],
            getattr(torch, indices["T"]),
        )
        self.chunk_num = indices["C"]

    gen_inputs = CallWorkload.gen_inputs


class NSATopKCall(CallWorkload, NSATopKWorkload):
    """A manifest call of NSATopKVarlenFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        NSATopKWorkload.__init__(
            self,
            indices["N"],
            indices["T_q"],
            indices["H"],
            indices["D"],
            indices["H"] // indices["H_kv"],
            params["sm_scale"],
            params["selected_block_num"],
            params["block_size"],
            getattr(torch, indices["T"]),
        )
        self.chunk_num = indices["C"]

    gen_inputs = CallWorkload.gen_inputs


class NSAFwdCall(CallWorkload, NSAFwdWorkload):
    """A manifest call of NSAVarlenFwdOp; the row's generators make the selection."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        NSAFwdWorkload.__init__(
            self,
            indices["N"],
            indices["H"],
            indices["T_q"],
            indices["D"],
            params["is_causal"],
            params["sm_scale"],
            params["block_size"],
            indices["H"] // indices["H_kv"],
            indices["SEL"],
            getattr(torch, indices["T"]),
        )
        # The reference scales the selected branch by its gate, which this op does not take.
        self.g_slc = torch.ones(
            (indices["N"], indices["T_q"], indices["H"]), dtype=self.dtype, device=run_device()
        )

    gen_inputs = CallWorkload.gen_inputs
