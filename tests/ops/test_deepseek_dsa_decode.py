import dataclasses

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, standard_tolerance
from tileops.kernels.attention import SparseMlaBasicKernel, SparseMlaCall
from tileops.kernels.attention.dsa import decode as dsa_decode
from tileops.ops import DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
from workloads.attention.dsa import DsaDecodeWorkload
from workloads.device import run_device


class DsaDecodeTest(DsaDecodeWorkload, TestBase):
    pass


class DsaDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, seq_len_q, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, "
            "q_start_index_s, sm_scale, dtype, tune",
            [
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    128,
                    1024,
                    2048,
                    512,
                    64,
                    2048,
                    1,
                    1,
                    1024,
                    None,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                    id="full-fp16-tuned",
                ),
            ],
        ),
    ]


@DsaDecodeFixture
def test_sparse_mla_decode(
    batch: int,
    heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    dim: int,
    dim_tail: int,
    topk: int,
    stride_kv: int,
    heads_kv: int,
    q_start_index_s: int,
    sm_scale: float,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = DsaDecodeTest(
        batch,
        heads,
        seq_len_q,
        seq_len_kv,
        dim,
        dim_tail,
        topk,
        stride_kv,
        heads_kv,
        q_start_index_s,
        sm_scale=sm_scale,
        dtype=dtype,
    )
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(
        dim_tail, stride_kv, q_start_index_s, sm_scale=sm_scale, tune=tune
    )
    test.check(op, *test.gen_inputs(), atol=3e-4, rtol=1e-5)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dim_tail", "dtype"),
    [
        pytest.param(64, torch.bfloat16, id="bf16-tail"),
        pytest.param(0, torch.bfloat16, id="bf16-no-tail"),
        pytest.param(0, torch.float16, id="fp16-no-tail"),
    ],
)
def test_sparse_mla_decode_tail_and_dtype(dim_tail, dtype) -> None:
    """BF16 preserves the output dtype; a zero tail omits the extra QK contraction."""
    test = DsaDecodeTest(1, 64, 7, 256, 512, dim_tail, 128, 1, 1, 256, dtype=dtype)
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(dim_tail, 1, 256)
    test.check(op, *test.gen_inputs(), **standard_tolerance(dtype))


def _padded_topk_indices(
    batch: int,
    seq_len: int,
    heads_kv: int,
    topk: int,
    seq_len_kv: int,
    pad: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """A top-k list that fills half its slots and pads the rest with *pad*."""
    indices = torch.full(
        (batch, seq_len, heads_kv, topk), pad, dtype=torch.int32, device=run_device()
    )
    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads_kv):
                selected = torch.randperm(seq_len_kv, generator=generator)[: topk // 2]
                indices[b, t, h, : topk // 2] = selected.to(torch.int32).to(run_device())
    return indices


@pytest.mark.smoke
def test_sparse_mla_decode_ignores_padded_topk_slots() -> None:
    """A slot no row fills must not reach kv, whatever value pads it.

    The cache here is shorter than the causal window, so the causal limit alone
    accepts a padding index; the kernel has to reject it on the kv extent.
    """
    batch, heads, seq_len, seq_len_kv = 1, 128, 64, 1024
    dim, dim_tail, topk, stride_kv, heads_kv, q_start = 512, 64, 128, 1, 1, 1024
    generator = torch.Generator().manual_seed(0)

    test = DsaDecodeTest(
        batch, heads, seq_len, seq_len_kv, dim, dim_tail, topk, stride_kv, heads_kv, q_start
    )
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(dim_tail, stride_kv, q_start)
    q, kv, _ = test.gen_inputs()

    # seq_len_kv is the padding the workloads write and the reference reads.
    in_range_pad = _padded_topk_indices(
        batch, seq_len, heads_kv, topk, seq_len_kv, seq_len_kv, generator
    )
    # The reference sums in float32 and rounds once; one fp16 ulp here is 1e-3.
    test.check(op, q, kv, in_range_pad, atol=2e-3, rtol=2e-3)

    expected = op(q, kv, in_range_pad)
    assert torch.isfinite(expected).all(), "an in-range padding slot produced a non-finite output"
    for pad in (-1, 2**30):
        padded = in_range_pad.clone()
        padded[padded == seq_len_kv] = pad
        assert torch.equal(op(q, kv, padded), expected), f"padding with {pad} changed the output"

    # Masked rows carry no state from one launch into the next: a launch that stages
    # NaN rows in every slot must not change the output.
    op(q, torch.full_like(kv, float("nan")), torch.zeros_like(in_range_pad))
    assert torch.equal(op(q, kv, in_range_pad), expected)


@pytest.mark.smoke
def test_sparse_mla_basic_refuses_what_its_shared_memory_cannot_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On SM89 the heads per block and the refusal follow the default config's shared memory,
    tuned or not: 16 heads at d=1024 with a 16-wide tail take 99,840 bytes over one KV tile
    (topk 32), served, and 101,888 over two (topk 64), refused."""
    call = SparseMlaCall(
        arch=89,
        sm_count=1,
        batch=1,
        seq_len=1,
        seq_len_kv=2048,
        heads=64,
        dim=512,
        tail_dim=64,
        dtype=torch.float16,
        topk=2048,
        kv_stride=1,
    )
    monkeypatch.setattr(SparseMlaBasicKernel, "_check_arch", lambda self: None)
    monkeypatch.setattr(dsa_decode, "get_sm_version", lambda index=None: 89)
    for tail_dim, block_h in ((64, 32), (512, 16)):
        assert SparseMlaBasicKernel.refusal(dataclasses.replace(call, tail_dim=tail_dim)) is None
        kernel = SparseMlaBasicKernel(1, 1, 2048, 64, 512, tail_dim, torch.float16, 2048, 1, 2047)
        assert kernel.config["block_h"] == block_h
    tight = dataclasses.replace(call, heads=16, dim=1024, tail_dim=16, topk=32)
    assert SparseMlaBasicKernel.refusal(tight) is None
    assert "101888" in SparseMlaBasicKernel.refusal(dataclasses.replace(tight, topk=64))
    assert SparseMlaBasicKernel.refusal(dataclasses.replace(call, arch=90)) is None
    for wide in (dataclasses.replace(call, dim=2048), dataclasses.replace(call, tail_dim=1024)):
        assert "shared memory" in SparseMlaBasicKernel.refusal(wide)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dim", "topk", "expected"),
    [
        pytest.param(512, 2048, "sparse_mla_kernel", id="ws"),
        pytest.param(512, 96, "sparse_mla_basic_kernel", id="topk-off-128"),
        pytest.param(64, 2048, "sparse_mla_basic_kernel", id="dim-off-128"),
    ],
)
def test_sparse_mla_regions(dim: int, topk: int, expected: str) -> None:
    """The warp-specialized kernel serves SM90 where its gather and topk tiling apply."""
    op = DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(64, 1, 0)
    call = SparseMlaCall(
        arch=90,
        sm_count=132,
        batch=1,
        seq_len=1,
        seq_len_kv=2048,
        heads=64,
        dim=dim,
        tail_dim=64,
        dtype=torch.float16,
        topk=topk,
        kv_stride=1,
    )
    assert op.select_implementation("sparse_mla", call) == expected
