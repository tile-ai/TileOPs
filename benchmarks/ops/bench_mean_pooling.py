from typing import Optional

import pytest
import torch

from tests.ops.test_mean_pooling_ops import MeanPoolingFixture, MeanPoolingTest
from tests.nsa_utils import prepare_chunk_indices
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MeanPoolingForwardOp


class MeanPoolingBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        # Mean pooling: sum chunk_size elements + divide, per output element
        return t.batch_size * t.chunks_per_bacth * t.heads * t.dim * t.chunk_size

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        # Read input + write output
        input_bytes = t.batch_size * t.seq_len * t.heads * t.dim * t.dtype.itemsize
        output_bytes = t.batch_size * t.chunks_per_bacth * t.heads * t.dim * t.dtype.itemsize
        return input_bytes + output_bytes


@MeanPoolingFixture
def test_mean_pooling_bench(batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int,
                            dtype: torch.dtype, accum_dtype: torch.dtype, tune: bool,
                            offsets: Optional[torch.Tensor]) -> None:
    if offsets is not None:
        assert batch_size == 1
        assert offsets[-1] == seq_len
        indices = prepare_chunk_indices(offsets, chunk_size)
        chunks_per_bacth = indices.shape[0]
        seq_num = offsets.shape[0] - 1
        use_offsets = 1
    else:
        offsets = torch.arange(
            0, (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device='cuda',
            requires_grad=False)
        chunks_per_bacth = (seq_len + chunk_size - 1) // chunk_size
        indices = torch.empty((chunks_per_bacth, 2), dtype=torch.int32, device='cuda')
        seq_num = batch_size
        use_offsets = 0

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "dim": dim,
        "chunk_size": chunk_size,
        "chunks_per_bacth": chunks_per_bacth,
        "seq_num": seq_num,
        "use_offsets": use_offsets,
        "dtype": dtype,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }

    test = MeanPoolingTest(
        batch_size=batch_size, seq_len=seq_len, heads=heads, dim=dim,
        chunk_size=chunk_size, chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num, use_offsets=use_offsets,
        dtype=dtype, accum_dtype=accum_dtype,
        offsets=offsets, indices=indices)

    bm = MeanPoolingBenchmark(test)
    inputs = test.gen_inputs()

    op = MeanPoolingForwardOp(**params)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record("mean_pooling", locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record("mean_pooling", locals(), result_bl, tag="baseline")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
