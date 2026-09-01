from typing import Optional

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import MeanPoolingForwardOp
from workloads.nsa_utils import prepare_chunk_indices
from workloads.pool import MeanPoolingWorkload


class MeanPoolingTest(MeanPoolingWorkload, TestBase):
    pass


class MeanPoolingFixture(FixtureBase):
    PARAMS = [
        # because of using warp reduction, the chunk_size must be divisible by 32
        (
            "batch_size, seq_len, heads, dim, chunk_size, dtype, accum_dtype, tune, offsets",
            [
                pytest.param(
                    1,
                    8192,
                    64,
                    128,
                    64,
                    torch.float16,
                    torch.float32,
                    False,
                    None,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    8192,
                    64,
                    128,
                    64,
                    torch.float16,
                    torch.float32,
                    True,
                    None,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    2048,
                    64,
                    128,
                    64,
                    torch.float16,
                    torch.float32,
                    False,
                    None,
                    marks=pytest.mark.full,
                ),
                # varlen case: lengths [256, 512, 256] -> offsets [0, 256, 768, 1024]
                pytest.param(
                    1,
                    1024,
                    64,
                    128,
                    64,
                    torch.float16,
                    torch.float32,
                    False,
                    torch.tensor([0, 256, 768, 1024], dtype=torch.int32, device="cuda"),
                    marks=pytest.mark.full,
                ),
                # varlen case: lengths [2048, 2048, 2048, 2048] -> offsets [0, 2048, 4096, 6144, 8192]
                pytest.param(
                    1,
                    8192,
                    64,
                    128,
                    64,
                    torch.float16,
                    torch.float32,
                    False,
                    torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device="cuda"),
                    marks=pytest.mark.full,
                ),
                # varlen case: lengths [100, 200, 300, 400] -> offsets [0, 100, 300, 600, 1000]
                pytest.param(
                    1,
                    1000,
                    64,
                    128,
                    32,
                    torch.float16,
                    torch.float32,
                    False,
                    torch.tensor([0, 100, 300, 600, 1000], dtype=torch.int32, device="cuda"),
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


@MeanPoolingFixture
def test_mean_pooling_op(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
    offsets: Optional[torch.Tensor],
) -> None:
    if offsets is not None:
        assert batch_size == 1
        assert offsets[-1] == seq_len
        indices = prepare_chunk_indices(offsets, chunk_size)
        chunks_per_batch = indices.shape[0]
        seq_num = offsets.shape[0] - 1
        use_offsets = 1
    else:
        offsets = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device="cuda",
            requires_grad=False,
        )
        chunks_per_batch = (seq_len + chunk_size - 1) // chunk_size  # integer ceil
        indices = torch.randint(0, seq_len, (chunks_per_batch, 2), dtype=torch.int32, device="cuda")
        seq_num = batch_size
        use_offsets = 0

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "dim": dim,
        "chunk_size": chunk_size,
        "chunks_per_batch": chunks_per_batch,
        "seq_num": seq_num,
        "use_offsets": use_offsets,
        "accum_dtype": accum_dtype,
        "tune": tune,
    }

    test = MeanPoolingTest(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        chunks_per_batch=chunks_per_batch,
        seq_num=seq_num,
        use_offsets=use_offsets,
        dtype=dtype,
        accum_dtype=accum_dtype,
        offsets=offsets,
        indices=indices,
    )

    op = MeanPoolingForwardOp(**params)
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-3, rtol=1e-5)


@pytest.mark.smoke
def test_mean_pooling_rejects_a_wrong_offsets_dtype() -> None:
    """The manifest's dtype contract has to be enforced on the call path.

    `forward` once dispatched a kernel without running the generated validator, so an
    `int64` offsets tensor reached TileLang instead of being rejected here.
    """
    op = MeanPoolingForwardOp(
        batch_size=1,
        seq_len=64,
        heads=1,
        dim=8,
        chunk_size=32,
        chunks_per_batch=2,
        seq_num=1,
        use_offsets=0,
        accum_dtype=torch.float32,
    )
    x = torch.randn(1, 64, 1, 8, device="cuda", dtype=torch.float16)
    offsets = torch.tensor([0, 64], dtype=torch.int64, device="cuda")
    indices = torch.zeros((2, 2), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="offsets"):
        op(x, offsets, indices)


@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(100, marks=pytest.mark.smoke),
        pytest.param(256, marks=pytest.mark.full),
    ],
)
def test_mean_pooling_dim_not_one_full_tile(dim: int) -> None:
    """`dim` values the manifest allows but no workload row carries.

    The kernel tiles `dim` by a width of at most 128: 100 is one tile with a short tail,
    256 is two whole tiles. Between them — 129 through 255 — TileLang finds no layout,
    which is why the manifest rules out that range rather than the op checking for it.
    """
    test = MeanPoolingTest(
        batch_size=1,
        seq_len=64,
        heads=2,
        dim=dim,
        chunk_size=32,
        chunks_per_batch=2,
        seq_num=1,
        use_offsets=0,
        dtype=torch.float16,
        accum_dtype=torch.float32,
    )
    op = MeanPoolingForwardOp(
        batch_size=1,
        seq_len=64,
        heads=2,
        dim=dim,
        chunk_size=32,
        chunks_per_batch=2,
        seq_num=1,
        use_offsets=0,
        accum_dtype=torch.float32,
    )
    test.check(op, *test.gen_inputs(), atol=1e-3, rtol=1e-5)
