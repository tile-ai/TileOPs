"""Test DeepSeek NSA GQA Window Sliding operation."""

from typing import Tuple

import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tileops.ops import GQAWindowSlidingOp


class GqaWindowSlidingFixture(FixtureBase):
    PARAMS = [
        ("batch_size, groups, uq, ukv, heads, dim, is_causal, window_size_left, "
         "window_size_right, dtype, accum_dtype, tune", [
             (1, 16, 1024, 1024, 64, 128, True, 32, -1, torch.float16, torch.float32, False),
             (3, 16, 8192, 8192, 64, 128, True, 2048, 0, torch.float16, torch.float32, False),
             (3, 16, 8192, 8192, 64, 128, False, -1, -1, torch.float16, torch.float32, False),
         ]),
    ]


class GqaWindowSlidingTest(TestBase):

    def __init__(self, batch_size: int, groups: int, uq: int, ukv: int, heads: int, dim: int,
                 is_causal: bool, window_size_left: int, window_size_right: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype) -> None:
        self.batch_size = batch_size
        self.groups = groups
        self.uq = uq
        self.ukv = ukv
        self.heads = heads
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
        rand_indices_q = torch.randperm(self.uq)[:self.batch_size - 1]
        cu_seqlens_q = torch.cat(
            [torch.tensor([0]),
             torch.arange(1, self.uq)[rand_indices_q],
             torch.tensor([self.uq])], 0).cuda().sort()[0].to(torch.int32)
        rand_indices_k = torch.randperm(self.ukv)[:self.batch_size - 1]
        cu_seqlens_k = torch.cat([
            torch.tensor([0]),
            torch.arange(1, self.ukv)[rand_indices_k],
            torch.tensor([self.ukv])
        ], 0).cuda().sort()[0].to(torch.int32)

        q = torch.randn((self.uq, self.heads, self.dim), dtype=self.dtype, device="cuda")
        k = torch.randn((self.ukv, self.heads // self.groups, self.dim),
                        dtype=self.dtype,
                        device="cuda")
        v = torch.randn((self.ukv, self.heads // self.groups, self.dim),
                        dtype=self.dtype,
                        device="cuda")
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    cu_seqlens_q: torch.LongTensor, cu_seqlens_k: torch.LongTensor,
                    max_seqlen_q: int) -> torch.Tensor:
        """PyTorch reference implementation for GQA window sliding attention (vectorized)"""
        device = q.device
        head_kv = self.heads // self.groups
        scale = (1.0 / self.dim)**0.5
        has_window = self.window_size_left >= 0 or self.window_size_right >= 0

        output = torch.zeros((self.uq, self.heads, self.dim), dtype=q.dtype, device=device)

        for batch_idx in range(self.batch_size):
            q_start = cu_seqlens_q[batch_idx].item()
            q_end = cu_seqlens_q[batch_idx + 1].item()
            kv_start = cu_seqlens_k[batch_idx].item()
            kv_end = cu_seqlens_k[batch_idx + 1].item()

            q_seqlen = q_end - q_start
            kv_seqlen = kv_end - kv_start

            if q_seqlen == 0:
                continue

            q_batch = q[q_start:q_end]
            k_batch = k[kv_start:kv_end]
            v_batch = v[kv_start:kv_end]

            offset = kv_seqlen - q_seqlen

            output_batch = torch.zeros((q_seqlen, self.heads, self.dim),
                                       dtype=q.dtype,
                                       device=device)

            for kv_head_idx in range(head_kv):
                head_start = kv_head_idx * self.groups
                head_end = head_start + self.groups

                q_group = q_batch[:, head_start:head_end, :]
                k_head = k_batch[:, kv_head_idx, :]
                v_head = v_batch[:, kv_head_idx, :]

                scores = torch.einsum('qgd,kd->qgk', q_group, k_head) * scale

                q_positions = torch.arange(q_seqlen, device=device, dtype=torch.float32)
                kv_positions = torch.arange(kv_seqlen, device=device, dtype=torch.float32)
                q_abs_positions = q_positions.unsqueeze(-1) + offset
                kv_abs_positions = kv_positions.unsqueeze(0)

                mask = torch.zeros((q_seqlen, kv_seqlen), dtype=torch.bool, device=device)

                if self.is_causal:
                    causal_mask = (q_positions.unsqueeze(-1) + offset < kv_positions.unsqueeze(0))
                    mask = mask | causal_mask

                if has_window:
                    if self.window_size_left >= 0:
                        window_left_mask = kv_abs_positions < (
                            q_abs_positions - self.window_size_left)
                        mask = mask | window_left_mask

                    if self.window_size_right >= 0:
                        window_right_mask = kv_abs_positions > (
                            q_abs_positions + self.window_size_right)
                        mask = mask | window_right_mask

                scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

                if self.is_causal and offset < 0:
                    invalid_mask = (q_positions + offset < 0)
                    scores = scores.masked_fill(
                        invalid_mask.unsqueeze(-1).unsqueeze(-1), float('-inf'))

                probs = torch.softmax(scores, dim=-1)

                out_group = torch.einsum('qgk,kd->qgd', probs, v_head)

                if self.is_causal and offset < 0:
                    invalid_positions = (q_positions + offset < 0)
                    out_group[invalid_positions] = 0

                output_batch[:, head_start:head_end, :] = out_group

            output[q_start:q_end] = output_batch

        return output


@GqaWindowSlidingFixture
def test_nsa_gqa_window_sliding_op(
    batch_size: int,
    groups: int,
    uq: int,
    ukv: int,
    heads: int,
    dim: int,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:
    assert groups % 16 == 0, "Group size must be a multiple of 16 in NSA"

    test = GqaWindowSlidingTest(batch_size, groups, uq, ukv, heads, dim, is_causal,
                                window_size_left, window_size_right, dtype, accum_dtype)
    params = {
        "batch_size": batch_size,
        "groups": groups,
        "uq": uq,
        "ukv": ukv,
        "heads": heads,
        "dim": dim,
        "is_causal": is_causal,
        "window_size_left": window_size_left,
        "window_size_right": window_size_right,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }
    op = GQAWindowSlidingOp(**params)
    test.check(op, *test.gen_inputs(), atol=3e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
