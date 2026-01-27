from typing import Optional

import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import MeanPoolingForwardBenchmark
from benchmarks.deepseek_nsa.utils import prepare_chunk_indices
from top.ops import MeanPoolingForwardOp


@pytest.mark.parametrize(
    # because of using warp reduction, the chunk_size must be divisible by 32
    "batch_size, seq_len, heads, dim, chunk_size, dtype, accum_dtype, tune, offsets",
    [
        (1, 8192, 64, 128, 64, torch.float16, torch.float32, False, None),
        (1, 8192, 64, 128, 64, torch.float16, torch.float32, True, None),
        (2, 2048, 64, 128, 64, torch.float16, torch.float32, False, None),
        # varlen case: lengths [256, 512, 256] -> offsets [0, 256, 768, 1024]
        (1, 1024, 64, 128, 64, torch.float16, torch.float32, False,
         torch.tensor([0, 256, 768, 1024], dtype=torch.int32, device='cuda')),
        # varlen case: lengths [2048, 2048, 2048, 2048] -> offsets [0, 2048, 4096, 6144, 8192]
        (1, 8192, 64, 128, 64, torch.float16, torch.float32, True,
         torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device='cuda')),
        # varlen case: lengths [100, 200, 300, 400] -> offsets [0, 100, 300, 600, 1000]
        (1, 1000, 64, 128, 32, torch.float16, torch.float32, True,
         torch.tensor([0, 100, 300, 600, 1000], dtype=torch.int32, device='cuda')),
    ],
)
def test_mean_pooling_op(batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int,
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
        chunks_per_bacth = (seq_len + chunk_size - 1) // chunk_size  # integer ceil
        indices = torch.randint(0, seq_len, (chunks_per_bacth, 2), dtype=torch.int32, device='cuda')
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

    op = MeanPoolingForwardOp(**params)

    benchmark = MeanPoolingForwardBenchmark(**params, offsets=offsets, indices=indices)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)


if __name__ == "__main__":
    test_mean_pooling_op(1, 8192, 64, 128, 64, torch.float16, torch.float32, False, None)
    test_mean_pooling_op(1, 8192, 64, 128, 64, torch.float16, torch.float32, True, None)
    test_mean_pooling_op(2, 2049, 64, 128, 64, torch.float16, torch.float32, False, None)
    test_mean_pooling_op(1, 1024, 64, 128, 64, torch.float16, torch.float32, False,
                         torch.tensor([0, 256, 768, 1024], dtype=torch.int32, device='cuda'))
    test_mean_pooling_op(
        1, 8192, 64, 128, 64, torch.float16, torch.float32, True,
        torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device='cuda'))
    test_mean_pooling_op(1, 1000, 64, 128, 32, torch.float16, torch.float32, True,
                         torch.tensor([0, 100, 300, 600, 1000], dtype=torch.int32, device='cuda'))
