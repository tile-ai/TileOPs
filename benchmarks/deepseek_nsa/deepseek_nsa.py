from typing import Any, Optional, Union

import torch
from einops import rearrange, repeat

from benchmarks.benchmark import Benchmark
from top.ops import MeanPoolingForwardOp, NSAFwdVarlenOp, NSATopkVarlenOp
from .utils import prepare_token_indices, prepare_chunk_offsets


class MeanPoolingForwardBenchmark(Benchmark):
    op_type = MeanPoolingForwardOp

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        chunks_per_bacth: int,
        seq_num: int,
        use_offsets: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunks_per_bacth = chunks_per_bacth
        self.seq_num = seq_num
        self.use_offsets = use_offsets
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.tune = tune
        # tilelang kernel needs offsets/indices to be provided
        self.offsets = offsets
        self.indices = indices

    @property
    def total_flops(self) -> int:
        return self.heads * self.dim * (self.seq_len + self.seq_num)

    @property
    def total_memory(self) -> int:
        return self.heads * self.dim * (self.seq_len +
                                        self.seq_num) * self.dtype.itemsize + 16 * self.seq_num

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return x, self.offsets, self.indices

    def ref_program(self, x: torch.Tensor, offsets: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        _ = indices
        batch_size, seq_len, heads, dim = x.shape

        if self.use_offsets == 0:
            output = torch.empty(
                batch_size, self.chunks_per_bacth, heads, dim, dtype=x.dtype, device=x.device)
            for chunk_id in range(self.chunks_per_bacth):
                start_token = chunk_id * self.chunk_size
                end_token = min(start_token + self.chunk_size, seq_len)
                output[:, chunk_id] = x[:, start_token:end_token].mean(dim=1)
        else:
            offsets = offsets.to(x.device)
            lengths = offsets[1:] - offsets[:-1]
            chunk_counts = ((lengths + self.chunk_size - 1) // self.chunk_size).tolist()
            total_chunks = sum(chunk_counts)
            output = torch.empty(
                batch_size, total_chunks, heads, dim, dtype=x.dtype, device=x.device)
            chunk_idx = 0
            for b in range(batch_size):
                for seq_id, chunks_i in enumerate(chunk_counts):
                    seq_start = offsets[seq_id].item()
                    seq_end = offsets[seq_id + 1].item()
                    for local_chunk_id in range(chunks_i):
                        chunk_start = seq_start + local_chunk_id * self.chunk_size
                        chunk_end = min(chunk_start + self.chunk_size, seq_end)
                        output[b, chunk_idx] = x[b, chunk_start:chunk_end].mean(dim=0)
                        chunk_idx += 1
        return output

    def baseline_program(self, x: torch.Tensor, offsets: torch.Tensor,
                         indices: torch.Tensor) -> torch.Tensor:
        from fla.ops.utils import mean_pooling
        _ = indices
        if self.use_offsets == 1:
            return mean_pooling(x, self.chunk_size, offsets, head_first=False)
        return mean_pooling(x, self.chunk_size, None, head_first=False)

    def baseline_profile(self,
                         *inputs: tuple[Any],
                         warmup: int = 100,
                         rep: int = 100,
                         device: str = "cuda:0") -> torch.Tensor:
        print("===== Profiling Mean Pooling Forward backend =====")
        return super().baseline_profile(
            self.baseline_program,
            *inputs,
            backend="mean_pooling_fwd",
            warmup=warmup,
            rep=rep,
            device=device)


class NSAFwdVarlenBenchmark(Benchmark):
    op_type = NSAFwdVarlenOp

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
        tune: bool = False,
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
        self.tune = tune

        self.head_kv = self.heads // self.groups

    @property
    def total_flops(self) -> int:
        flops_per_token = 4 * self.dim * self.selected_blocks * self.block_size
        return flops_per_token * self.c_seq_len * self.heads

    @property
    def total_memory(self) -> int:
        # q, k, v, output, block_indices, block_counts, offsets, token_indices
        # ignore block counts, offsets and token_indices memory
        q_memory = self.heads * self.c_seq_len * self.dim * self.dtype.itemsize
        k_memory = self.head_kv * self.c_seq_len * self.dim * self.dtype.itemsize
        v_memory = self.head_kv * self.c_seq_len * self.dim * self.dtype.itemsize
        output_memory = self.heads * self.c_seq_len * self.dim * self.dtype.itemsize
        block_indices_memory = self.head_kv * self.c_seq_len * self.selected_blocks * 4
        return (q_memory + k_memory + v_memory + output_memory + block_indices_memory)

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

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    block_indices: torch.Tensor, block_counts: torch.Tensor, offsets: torch.Tensor,
                    token_indices: torch.Tensor) -> torch.Tensor:
        _ = token_indices
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        block_indices = block_indices.unsqueeze(0)
        block_counts = block_counts.unsqueeze(0)
        return self.naive_nsa(
            q=q,
            k=k,
            v=v,
            g_slc=self.g_slc,
            g_swa=self.g_swa,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=self.block_size,
            window_size=0,
            scale=self.scale,
            cu_seqlens=offsets,
            head_first=False,
        )

    def baseline_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         block_indices: torch.Tensor, block_counts: torch.Tensor,
                         offsets: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        from native_sparse_attention.ops.parallel import parallel_nsa_fwd
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        block_indices = block_indices.unsqueeze(0)
        block_counts = block_counts.unsqueeze(0)
        return parallel_nsa_fwd(q, k, v, block_indices, block_counts, self.block_size, self.scale,
                                offsets, token_indices)

    def baseline_profile(
        self,
        *inputs: tuple[torch.Tensor, ...],
        warmup: int = 100,
        rep: int = 100,
        device: str = "cuda:0",
    ) -> torch.Tensor:
        print("===== Profiling FLA NSA_Fwd backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FLA", warmup=warmup, rep=rep, device=device)


class NSATopkVarlenBenchmark(Benchmark):
    op_type = NSATopkVarlenOp

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
        bk: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        tune: bool = False,
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
        self.bk = bk
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.tune = tune

        self.head_kv = self.heads // self.group

    @property
    def total_flops(self) -> int:
        # Step 1 (LSE) + Step 2 (Scores)
        # Total: c_seq_len * head_kv * 2 * (2 * group * dim * (c_seq_len / (2 * bs)))
        return (2 * self.heads * self.dim * self.c_seq_len**2) // self.bs

    @property
    def total_memory(self) -> int:
        # q: read once, k_cmp: read twice per preceding block per token, block_indices: write once
        q_read = self.heads * self.c_seq_len * self.dim * self.dtype.itemsize
        k_read = (self.head_kv * self.dim * self.c_seq_len**2 * self.dtype.itemsize) // self.bs
        indices_write = self.c_seq_len * self.head_kv * self.selected_block_num * 4
        return q_read + k_read + indices_write

    def check(self, op: Any, *inputs: Any, threshold: float = 1e-3) -> None:
        outputs_ref = self.ref_program(*inputs)
        outputs = op(*inputs)

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for out, ref in zip(outputs, outputs_ref):
            print("[Top-K Indices Comparison - TileLang vs PyTorch]")

            indices_match_tl_torch = torch.all(out == ref)
            if indices_match_tl_torch:
                print("✅ Top-K Indices Matched!")
            else:
                mismatch_count = (out != ref).sum().item()
                total_count = out.numel()
                mismatch_ratio = mismatch_count / total_count

                assert mismatch_ratio <= threshold, \
                    f"Top-K mismatch ratio {mismatch_ratio:.3%} exceeds threshold {threshold:.3%}"
                print(f"⚠️ Top-K Indices Mismatched slightly within threshold: \
                        {mismatch_ratio * 100:.3f}%")
        print(f"All checks passed for {op.__class__.__name__}.✅")

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
        chunk_num = chunk_offsets[-1].item()

        # float16, data Tie-breaking
        q = torch.randn(
            (self.c_seq_len, self.heads, self.dim), dtype=self.dtype, device="cuda") * 0.1
        k = torch.randn((chunk_num, self.head_kv, self.dim), dtype=self.dtype, device="cuda") * 0.1

        q.requires_grad_(True)
        k.requires_grad_(True)

        lse = torch.zeros((self.c_seq_len, self.heads), dtype=self.dtype, device="cuda")
        token_indices = prepare_token_indices(offsets)
        chunk_offsets = prepare_chunk_offsets(offsets, self.bs)

        self.chunk_num = chunk_offsets[-1].item()
        return (
            q,
            k,
            lse,
            offsets.to(torch.int32),
            chunk_offsets.to(torch.int32),
            token_indices.to(torch.int32),
        )

    def nsa_topk_torch(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse: torch.Tensor,
        block_counts: int,
        block_size: int,
        scale: float,
        offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
    ) -> torch.Tensor:
        _ = lse
        q = q.squeeze(0) if q.dim() == 4 else q
        k_cmp = k_cmp.squeeze(0) if k_cmp.dim() == 4 else k_cmp
        c_seq_len, heads, dim = q.shape
        head_kv = k_cmp.shape[1]
        group = heads // head_kv
        selected_block_num = block_counts if isinstance(block_counts,
                                                        int) else block_counts.max().item()
        selected_block_num = 1 << (selected_block_num - 1).bit_length()
        bs = block_size
        scale_log2 = scale * 1.44269504

        device = q.device
        accum_dtype = torch.float32

        lse_out = torch.zeros((c_seq_len, heads), dtype=accum_dtype, device=device)
        block_indices = torch.zeros((c_seq_len, head_kv, selected_block_num),
                                    dtype=torch.int32,
                                    device=device)

        for i_c in range(c_seq_len):
            i_n, i_t = token_indices[i_c, 0].item(), token_indices[i_c, 1].item()
            bos = offsets[i_n].item()
            boc = chunk_offsets[i_n].item()
            nc = (i_t + 1) // bs

            q_curr = q[bos + i_t]

            for i_h in range(head_kv):
                q_h = q_curr[i_h * group:(i_h + 1) * group]
                scores_max = torch.full((group,), float('-inf'), dtype=accum_dtype, device=device)
                logsum = torch.zeros((group,), dtype=accum_dtype, device=device)

                for i_loop in range(0, nc, bs):
                    start_idx = i_loop
                    end_idx = min(start_idx + bs, nc)
                    curr_bc = end_idx - start_idx
                    k_blocks = k_cmp[boc + start_idx:boc + end_idx, i_h]
                    acc_s = torch.matmul(q_h.to(torch.float16),
                                         k_blocks.to(torch.float16).t()).to(accum_dtype)

                    if curr_bc < bs:
                        padding = torch.full((group, bs - curr_bc),
                                             float('-inf'),
                                             dtype=accum_dtype,
                                             device=device)
                        acc_s = torch.cat([acc_s, padding], dim=1)

                    o_c = torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device)
                    valid_mask = o_c < nc
                    acc_s = torch.where(
                        valid_mask.unsqueeze(0), acc_s, torch.full_like(acc_s, float('-inf')))

                    scores_max_prev = scores_max.clone()
                    scores_max_curr = acc_s.max(dim=1)[0]
                    scores_max = torch.maximum(scores_max, scores_max_curr)

                    scores_scale = torch.exp2((scores_max_prev - scores_max) * scale_log2)
                    acc_s_exp = torch.exp2((acc_s - scores_max.unsqueeze(1)) * scale_log2)
                    acc_s_exp = torch.where(acc_s > float('-inf'), acc_s_exp,
                                            torch.zeros_like(acc_s_exp))
                    logsum = logsum * scores_scale + acc_s_exp.sum(dim=1)

                if nc == 0:
                    b_lse = torch.zeros((group,), dtype=accum_dtype, device=device)
                else:
                    logsum_log2 = torch.where(
                        logsum > 0, torch.log2(logsum),
                        torch.full((group,), float('-inf'), dtype=accum_dtype, device=device))
                    b_lse = (scores_max * scale_log2 + logsum_log2) / 1.44269504
                    b_lse = torch.where(logsum <= 0, torch.zeros_like(b_lse), b_lse)
                lse_out[bos + i_t, i_h * group:(i_h + 1) * group] = b_lse

                nc_topk = i_t // bs + 1
                pool_scores = torch.full((bs * 2,), float('-inf'), dtype=accum_dtype, device=device)
                pool_indices = torch.zeros((bs * 2,), dtype=torch.int32, device=device)

                for i_tk in range(0, nc_topk, bs):
                    start_idx = i_tk
                    end_idx = min(start_idx + bs, nc_topk)
                    curr_bc_tk = end_idx - start_idx
                    k_blocks = k_cmp[boc + start_idx:boc + end_idx, i_h]
                    acc_s = torch.matmul(q_h.to(torch.float16),
                                         k_blocks.to(torch.float16).t()).to(accum_dtype)

                    if curr_bc_tk < bs:
                        padding = torch.full((group, bs - curr_bc_tk),
                                             float('-inf'),
                                             dtype=accum_dtype,
                                             device=device)
                        acc_s = torch.cat([acc_s, padding], dim=1)

                    o_c = torch.arange(start_idx, start_idx + bs, dtype=torch.int32, device=device)
                    is_curr = (o_c == i_t // bs)
                    is_hist = (o_c < i_t // bs)
                    importance = torch.where(
                        is_curr.unsqueeze(0),
                        torch.ones((group, bs), dtype=accum_dtype, device=device),
                        torch.where(
                            is_hist.unsqueeze(0),
                            torch.exp2((acc_s * scale - b_lse.unsqueeze(1)) * 1.44269504),
                            torch.zeros((group, bs), dtype=accum_dtype, device=device)))

                    b_i_current = importance.sum(dim=0)
                    pool_scores[bs:bs + bs] = b_i_current
                    pool_indices[bs:bs + bs] = torch.arange(
                        start_idx, start_idx + bs, dtype=torch.int32, device=device) + 1

                    o_c_valid = torch.arange(
                        start_idx, start_idx + bs, dtype=torch.int32, device=device) < nc_topk
                    pool_scores[bs:bs + bs] = torch.where(
                        o_c_valid, pool_scores[bs:bs + bs],
                        torch.full_like(pool_scores[bs:bs + bs], float('-inf')))
                    pool_indices[bs:bs + bs] = torch.where(
                        o_c_valid, pool_indices[bs:bs + bs],
                        torch.zeros_like(pool_indices[bs:bs + bs]))

                    eps, score_scale = 1e-5, 1e12
                    scores_quantized = (pool_scores / eps).round() * eps
                    sort_key = scores_quantized.to(torch.float64) * score_scale + pool_indices.to(
                        torch.float64)
                    sort_key = torch.where(
                        pool_indices > 0, sort_key,
                        torch.full_like(sort_key, float('-inf'), dtype=torch.float64))
                    sorted_indices = torch.argsort(sort_key, descending=True)

                    pool_scores = pool_scores[sorted_indices]
                    pool_indices = pool_indices[sorted_indices]

                final_indices = pool_indices[:selected_block_num] - 1
                final_indices = torch.where(final_indices >= 0, final_indices,
                                            torch.tensor(-1, dtype=torch.int32, device=device))
                block_indices[i_c, i_h, :selected_block_num] = final_indices.to(torch.int32)

        return block_indices

    def ref_program(
        self,
        q: torch.Tensor,
        k_cmp: torch.Tensor,
        lse: torch.Tensor,
        offsets: torch.LongTensor,
        chunk_offsets: torch.LongTensor,
        token_indices: torch.LongTensor,
    ) -> torch.Tensor:
        return self.nsa_topk_torch(q, k_cmp, lse, self.selected_block_num, self.bs, self.scale,
                                   offsets, token_indices, chunk_offsets)

    def baseline_program(self, q: torch.Tensor, k_cmp: torch.Tensor, lse: torch.Tensor,
                         offsets: torch.LongTensor, chunk_offsets: torch.LongTensor,
                         token_indices: torch.LongTensor) -> torch.Tensor:
        from native_sparse_attention.ops.parallel import parallel_nsa_topk
        q = q.unsqueeze(0)
        k_cmp = k_cmp.unsqueeze(0)
        lse = lse.unsqueeze(0)
        _, _ = chunk_offsets, token_indices
        return parallel_nsa_topk(q, k_cmp, None, self.bc, self.bs, self.scale, offsets)

    def baseline_profile(
        self,
        *inputs: tuple[torch.Tensor, ...],
        warmup: int = 100,
        rep: int = 100,
        device: str = "cuda",
    ) -> torch.Tensor:
        print("===== Profiling FLA NSA_Topk backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FLA", warmup=warmup, rep=rep, device=device)
