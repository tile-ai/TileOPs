from typing import Any, Optional

import torch
from fla.ops.utils import mean_pooling

from benchmarks.benchmark import Benchmark
from top.ops import MeanPoolingForwardOp


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
    def total_memory(self) -> int:  # noqa: U100
        return self.heads * self.dim * (self.seq_len +
                                        self.seq_num) * self.dtype.itemsize + 16 * self.seq_num

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return x, self.offsets, self.indices

    def ref_program(self, x: torch.Tensor, offsets: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        _ = indices
        if self.use_offsets == 1:
            return mean_pooling(x, self.chunk_size, offsets, head_first=False)
        return mean_pooling(x, self.chunk_size, None, head_first=False)

    def baseline_program(self, x: torch.Tensor, offsets: torch.Tensor,
                         indices: torch.Tensor) -> torch.Tensor:
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
