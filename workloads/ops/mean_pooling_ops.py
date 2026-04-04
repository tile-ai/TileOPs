from typing import Optional, Tuple

import torch

from workloads.base import WorkloadBase


class MeanPoolingTest(WorkloadBase):

    def __init__(self, batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int,
                 chunks_per_bacth: int, seq_num: int, use_offsets: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype,
                 offsets: Optional[torch.Tensor] = None,
                 indices: Optional[torch.Tensor] = None):
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
        self.offsets = offsets
        self.indices = indices

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.heads, self.dim,
            device='cuda', dtype=self.dtype)
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
