from typing import Tuple

import torch

from workloads.base import WorkloadBase
from workloads.nsa_utils import prepare_chunk_offsets, prepare_token_indices


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
