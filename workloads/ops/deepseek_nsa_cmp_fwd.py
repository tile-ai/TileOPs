from typing import Tuple

import torch

from workloads.base import WorkloadBase
from workloads.nsa_utils import prepare_chunk_offsets, prepare_token_indices


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

    def gen_inputs(self) -> Tuple[torch.Tensor, ...]:
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
