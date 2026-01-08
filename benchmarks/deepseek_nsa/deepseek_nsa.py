from typing import Any, Tuple

import torch
from fla.ops.common.utils import prepare_chunk_indices
from fla.ops.utils import mean_pooling

from benchmarks.benchmark import Benchmark
from top.ops import MeanPoolingForwardOp
from top.utils import str2dtype


class MeanPoolingForwardBenchmark(Benchmark):
    op_type = MeanPoolingForwardOp

    def __init__(self, batch_size, total_seqlen, total_chunks, heads, dim, chunk_size, tune=True):
        self.batch_size = batch_size
        self.total_seqlen = total_seqlen
        self.total_chunks = total_chunks
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.tune = tune
        self.dtype = torch.float16

    @property
    def total_flops(self):
        flops = self.heads * self.dim * (self.total_seqlen + self.total_chunks)
        return flops

    @property
    def total_memory(self):
        return self.heads * self.dim * (
            self.total_seqlen + self.total_chunks) * self.dtype.itemsize + 16 * self.total_chunks

    def gen_inputs(self):
        x_unpad = torch.randn(self.total_seqlen,
                              self.heads,
                              self.dim,
                              device='cuda',
                              dtype=self.dtype)
        # fixed length
        b = self.batch_size
        t = self.total_seqlen // b

        cu_seqlens = torch.arange(0, (b + 1) * t, t, dtype=torch.int32, device='cuda')
        chunk_indices = prepare_chunk_indices(cu_seqlens, self.chunk_size)

        return x_unpad, cu_seqlens, chunk_indices

    def ref_program(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor,
                    chunk_indices: torch.Tensor) -> torch.Tensor:
        b = self.batch_size
        t = self.total_seqlen // b
        x = x_unpad.view(b, t, self.heads, self.dim)

        return mean_pooling(x, chunk_size=self.chunk_size, cu_seqlens=None,
                            head_first=False).view(-1, self.heads, self.dim)

    def baseline_program(self, x_unpad: torch.Tensor, cu_seqlens: torch.Tensor,
                         chunk_indices: torch.Tensor) -> torch.Tensor:
        b = self.batch_size
        t = self.total_seqlen // b
        x = x_unpad.view(b, t, self.heads, self.dim)
        return mean_pooling(x, chunk_size=self.chunk_size, cu_seqlens=None,
                            head_first=False).view(-1, self.heads, self.dim)

    def baseline_profile(self,
                         *inputs: Any,
                         warmup: int = 100,
                         rep: int = 100,
                         device: str = "cuda:0") -> Any:
        print("===== Profiling Mean Pooling_Fwd backend =====")
        return super().baseline_profile(self.baseline_program,
                                        *inputs,
                                        backend="Mean Pooling",
                                        warmup=warmup,
                                        rep=rep,
                                        device=device)

