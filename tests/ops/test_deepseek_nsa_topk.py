from typing import Tuple

import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tests.nsa_utils import prepare_chunk_offsets, prepare_token_indices
from tileops.ops import NSATopkVarlenOp


class NsaTopkFixture(FixtureBase):
    PARAMS = [
        ("seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs, bk, "
         "dtype, accum_dtype, tune", [
             (5, 1024, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False),
             (3, 512, 32, 128, 16, 1, 16, 32, 32, 128, torch.float16, torch.float32, False),
         ]),
    ]


class NsaTopkTest(TestBase):

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

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
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
        bs = block_size
        LOG2_E = 1.44269504
        scale_log2 = scale * LOG2_E

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
                    acc_s = torch.matmul(q_h, k_blocks.t()).to(accum_dtype)

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
                    b_lse = (scores_max * scale_log2 + logsum_log2) / LOG2_E
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
                    acc_s = torch.matmul(q_h, k_blocks.t()).to(accum_dtype)

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
                            torch.exp2((acc_s * scale - b_lse.unsqueeze(1)) * LOG2_E),
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

    def check_topk(self, op, *inputs, threshold: float = 1e-3) -> None:
        """Custom check for topk indices (not floating point closeness)."""
        outputs_ref = self.ref_program(*inputs)
        outputs = op(*inputs)

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        for out, ref in zip(outputs, outputs_ref, strict=True):
            print("[Top-K Indices Comparison - TileLang vs PyTorch]")

            indices_match = torch.all(out == ref)
            if indices_match:
                print("Top-K Indices Matched!")
            else:
                mismatch_count = (out != ref).sum().item()
                total_count = out.numel()
                mismatch_ratio = mismatch_count / total_count

                assert mismatch_ratio <= threshold, \
                    f"Top-K mismatch ratio {mismatch_ratio:.3%} exceeds threshold {threshold:.3%}"
                print(f"Top-K Indices Mismatched slightly within threshold: "
                      f"{mismatch_ratio * 100:.3f}%")
        print(f"All checks passed for {op.__class__.__name__}.")


@NsaTopkFixture
def test_nsa_topk_varlen_op(
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
    tune: bool,
) -> None:
    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = NsaTopkTest(seq_num, c_seq_len, heads, dim, group, scale, selected_block_num, bc, bs,
                       bk, dtype, accum_dtype)
    inputs = test.gen_inputs()
    op = NSATopkVarlenOp(
        seq_num=seq_num, c_seq_len=c_seq_len, heads=heads, dim=dim, group=group, scale=scale,
        selected_block_num=selected_block_num, bc=bc, bs=bs, bk=bk, dtype=dtype,
        accum_dtype=accum_dtype, tune=tune, chunk_num=test.chunk_num)
    test.check_topk(op, *inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
