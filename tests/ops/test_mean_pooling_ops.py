from typing import Optional, Tuple

import pytest
import torch

from tests.test_base import TestBase, FixtureBase
from tests.nsa_utils import prepare_chunk_indices
from tileops.ops import MeanPoolingForwardOp


class MeanPoolingFixture(FixtureBase):
    PARAMS = [
        # because of using warp reduction, the chunk_size must be divisible by 32
        ("batch_size, seq_len, heads, dim, chunk_size, dtype, accum_dtype, tune, offsets", [
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
        ]),
    ]


class MeanPoolingTest(TestBase):

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


@MeanPoolingFixture
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

    test = MeanPoolingTest(
        batch_size=batch_size, seq_len=seq_len, heads=heads, dim=dim,
        chunk_size=chunk_size, chunks_per_bacth=chunks_per_bacth,
        seq_num=seq_num, use_offsets=use_offsets,
        dtype=dtype, accum_dtype=accum_dtype,
        offsets=offsets, indices=indices)

    op = MeanPoolingForwardOp(**params)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
