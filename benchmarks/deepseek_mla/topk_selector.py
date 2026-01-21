from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import TopkSelectorOp

class TopkSelectorBenchmark(Benchmark):
    op_type = TopkSelectorOp

    def __init__(self,
                batch: int,
                seq_len: int,
                topk: int,
                in_dtype: str,
                out_dtype: str,
                # index_score: torch.float32,
                # index: torch.int32,
                # starts: torch.int32,
                # ends: torch.int32
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        # self.index_score = index_score
        # self.index = index
        # self.starts = starts
        # self.ends = ends

    
    @property
    def total_flops(self) -> float:
        return None
    
    @property
    def total_memory(self) -> float:
        # index_score: batch, seq_len
        # index: batch, topk
        # starts: batch
        # ends: batch
        index_score_memory = self.batch * self.seq_len * self.in_dtype.itemsize
        index_memory = self.batch * self.topk * self.out_dtype.itemsize
        starts_memory = self.batch * self.out_dtype.itemsize
        ends_memory = self.batch * self.out_dtype.itemsize
        return index_score_memory + index_memory + starts_memory + ends_memory



    def gen_inputs(self) ->Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(
            self.batch, self.seq_len, dtype=self.in_dtype, device= "cuda"
        )
        starts = torch.zeros(self.batch, dtype=self.out_dtype, device = "cuda")
        ends = torch.ones(self.batch, dtype=self.out_dtype, device = "cuda") * self.seq_len
        return index_score, starts, ends


    def ref_program(self, index_score, staets, ends):
        indexes_ref = torch.topk(index_score, self.topk, dim=-1)[1]
        return indexes_ref
