from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.ops import MeanPoolingForwardOp
from workloads.attention.mean_pooling import MeanPoolingTest
from workloads.nsa_utils import prepare_chunk_indices


class _MeanPoolingTestBaseline(MeanPoolingTest):
    """Adds baseline ref_program for benchmark profiling."""

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


class MeanPoolingBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # Mean pooling: sum chunk_size elements + divide, per output element
        return t.batch_size * t.chunks_per_bacth * t.heads * t.dim * t.chunk_size

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # Read input + write output
        input_bytes = t.batch_size * t.seq_len * t.heads * t.dim * t.dtype.itemsize
        output_bytes = t.batch_size * t.chunks_per_bacth * t.heads * t.dim * t.dtype.itemsize
        return input_bytes + output_bytes


_MEAN_POOLING_BENCH_PARAMS = [
    pytest.param(1, 8192, 64, 128, 64, torch.float16, torch.float32, True, None, id="dense-mainstream"),
    pytest.param(2, 2048, 64, 128, 64, torch.float16, torch.float32, True, None, id="dense-batched"),
    pytest.param(
        1, 8192, 64, 128, 64, torch.float16, torch.float32, True,
        torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device="cuda"),
        id="varlen-long",
    ),
    pytest.param(
        1, 1000, 64, 128, 32, torch.float16, torch.float32, True,
        torch.tensor([0, 100, 300, 600, 1000], dtype=torch.int32, device="cuda"),
        id="varlen-tail",
    ),
]


@pytest.mark.parametrize(
    "batch_size, seq_len, heads, dim, chunk_size, dtype, accum_dtype, tune, offsets",
    _MEAN_POOLING_BENCH_PARAMS,
)
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

    test = _MeanPoolingTestBaseline(
        batch_size=batch_size, seq_len=seq_len, heads=heads, dim=dim,
        chunk_size=chunk_size, chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num, use_offsets=use_offsets,
        dtype=dtype, accum_dtype=accum_dtype,
        offsets=offsets, indices=indices)

    bm = MeanPoolingBenchmark(test)
    inputs = test.gen_inputs()

    op = MeanPoolingForwardOp(**params)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
