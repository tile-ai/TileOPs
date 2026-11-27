import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.mamba.mamba2_fwd import Mamba2FwdOp
from tileops._ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops._ops.mamba.ssd_chunk_cumsum import SSDChunkCumsumFwdOp
from tileops._ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops._ops.mamba.ssd_chunk_state import SSDChunkStateFwdOp
from tileops._ops.mamba.ssd_recurrent import SSDRecurrentFwdOp
from tileops._ops.mamba.ssd_state_passing import SSDStatePassingFwdOp
from workloads.device import run_device
from workloads.mamba import (
    SSDChunkCumsumFwdWorkload,
    SSDChunkScanFwdWorkload,
    SSDChunkStateFwdWorkload,
    SSDDecodeWorkload,
    SSDStatePassingFwdWorkload,
    SSDStatePassingUnitWorkload,
    coupling_verification,
    mamba2_fwd_ref,
    mamba2_verification,
    ssd_chunk_coupling_fwd_ref,
    ssd_chunk_state_fwd_ref,
    ssd_decode_result,
)
from workloads.numerics import compare_outputs


class SSDChunkCumsumFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype",
            [
                # feature: no bias, no softplus (baseline path)
                pytest.param(1, 2, 64, 4, False, False, torch.float16, marks=pytest.mark.smoke),
                # feature: bias only (has_dt_bias branch, no softplus)
                pytest.param(1, 2, 64, 4, True, False, torch.bfloat16, marks=pytest.mark.smoke),
                # feature: softplus only (no bias, dt_softplus branch)
                pytest.param(1, 2, 64, 4, False, True, torch.float16, marks=pytest.mark.smoke),
                # feature: bias + softplus (full pipeline)
                pytest.param(1, 2, 64, 4, True, True, torch.bfloat16, marks=pytest.mark.full),
                # shape: larger batch and chunk count
                pytest.param(2, 4, 64, 8, False, False, torch.float16, marks=pytest.mark.full),
                # shape: larger chunk_len tile
                pytest.param(1, 2, 128, 4, False, False, torch.bfloat16, marks=pytest.mark.full),
                # shape + feature: large shape with full pipeline
                pytest.param(2, 4, 128, 16, True, True, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class SSDChunkScanFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype",
            [
                pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 64, 8, 64, 64, 2, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class SSDChunkStateFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx",
            [
                pytest.param(
                    1,
                    2,
                    64,
                    4,
                    64,
                    32,
                    1,
                    torch.float16,
                    False,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    2,
                    128,
                    4,
                    128,
                    32,
                    1,
                    torch.bfloat16,
                    False,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2,
                    4,
                    64,
                    8,
                    64,
                    64,
                    2,
                    torch.float16,
                    False,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    2,
                    64,
                    4,
                    64,
                    32,
                    2,
                    torch.bfloat16,
                    False,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    4,
                    64,
                    8,
                    64,
                    64,
                    2,
                    torch.float16,
                    False,
                    True,
                    marks=pytest.mark.full,
                ),
                # A d_state that is not a whole number of 16-column GEMM steps.
                pytest.param(
                    1,
                    2,
                    64,
                    4,
                    64,
                    24,
                    1,
                    torch.float16,
                    False,
                    False,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class SSDDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, n_heads, d_head, d_state, n_groups, dtype, tune",
            [
                pytest.param(
                    1,
                    4,
                    64,
                    16,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    1,
                    4,
                    64,
                    16,
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2,
                    8,
                    64,
                    32,
                    2,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    2,
                    8,
                    128,
                    64,
                    4,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                ),
                # A lane takes two 16-byte pieces of each row, and the last CTA of a head
                # holds a partial row tile.
                pytest.param(
                    2,
                    4,
                    36,
                    256,
                    2,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                # An odd d_state: a lane takes single elements, and the second sweep of each
                # row holds one element, on lane 0.
                pytest.param(
                    2,
                    4,
                    64,
                    33,
                    1,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                ),
                # d_state 1: one lane a row, so its y needs no shuffle.
                pytest.param(
                    1,
                    4,
                    64,
                    1,
                    1,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                # Tuned: the op runs the config the tuner picks from the kernel's candidates.
                pytest.param(
                    1,
                    4,
                    64,
                    16,
                    1,
                    torch.float16,
                    True,
                    marks=pytest.mark.full,
                ),
            ],
        ),
    ]


class SSDStatePassingFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, num_chunks, n_heads, d_state, dtype",
            [
                pytest.param(1, 2, 4, 32, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 2, 4, 32, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 64, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 64, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune",
    [
        pytest.param(
            1,
            2,
            64,
            1,
            64,
            torch.float16,
            False,
            marks=[pytest.mark.smoke, pytest.mark.packaging(family="mamba")],
        ),
        pytest.param(1, 2, 64, 1, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
        pytest.param(1, 2, 64, 1, 64, torch.float16, True, marks=pytest.mark.full),
        pytest.param(1, 2, 64, 2, 64, torch.float16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 64, 1, 96, torch.float16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 128, 1, 64, torch.bfloat16, False, marks=pytest.mark.full),
        pytest.param(1, 2, 256, 1, 64, torch.float16, False, marks=pytest.mark.full),
        pytest.param(2, 4, 64, 4, 128, torch.bfloat16, False, marks=pytest.mark.full),
    ],
)
def test_ssd_chunk_coupling_fwd(batch, num_chunks, chunk_len, n_groups, d_state, dtype, tune):
    op = SSDChunkCouplingFwdOp(chunk_len)
    if tune:
        op.request_tune()
    seq_len = num_chunks * chunk_len
    C_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device=run_device()) * 0.1
    B_mat = torch.randn(batch, seq_len, n_groups, d_state, dtype=dtype, device=run_device()) * 0.1
    ref = ssd_chunk_coupling_fwd_ref(C_mat, B_mat, num_chunks, chunk_len, dtype)
    out = op(C_mat, B_mat)
    compare_outputs(out, ref, coupling_verification())
    # A cached executable must keep the same output allocation contract on later calls.
    next_c = -C_mat
    next_ref = ssd_chunk_coupling_fwd_ref(next_c, B_mat, num_chunks, chunk_len, dtype)
    compare_outputs(op(next_c, B_mat), next_ref, coupling_verification())
    compare_outputs(out, ref, coupling_verification())


@pytest.mark.smoke
def test_ssd_chunk_coupling_fwd_noncontiguous():
    """SSDChunkCouplingFwdOp must handle non-contiguous inputs."""
    batch, num_chunks, chunk_len, n_groups, d_state = 1, 2, 64, 1, 64
    dtype = torch.float16
    seq_len = num_chunks * chunk_len
    C_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=run_device())
    B_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=run_device())
    C_mat = C_full[:, ::2, :, :]
    B_mat = B_full[:, ::2, :, :]
    assert not C_mat.is_contiguous()
    assert not B_mat.is_contiguous()
    ref = ssd_chunk_coupling_fwd_ref(
        C_mat.contiguous(), B_mat.contiguous(), num_chunks, chunk_len, dtype
    )
    out = SSDChunkCouplingFwdOp(chunk_len)(C_mat, B_mat)
    compare_outputs(out, ref, coupling_verification())


class SSDChunkCumsumFwdTest(SSDChunkCumsumFwdWorkload, TestBase):
    pass


@SSDChunkCumsumFwdFixture
def test_ssd_chunk_cumsum_fwd(
    batch, num_chunks, chunk_len, n_heads, has_dt_bias, dt_softplus, dtype
):
    test = SSDChunkCumsumFwdTest(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        has_dt_bias=has_dt_bias,
        dt_softplus=dt_softplus,
        dtype=dtype,
    )
    op = SSDChunkCumsumFwdOp(
        chunk_size=chunk_len,
        dt_softplus=dt_softplus,
        out_dtype=dtype,
    )
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
def test_ssd_chunk_cumsum_fwd_padded_head_tile():
    """Five heads against block_h=4 is the only shape reaching the masked tail."""
    batch, n_heads, chunk_len, num_chunks = 1, 5, 64, 2
    seq_len = chunk_len * num_chunks
    op = SSDChunkCumsumFwdOp(chunk_size=chunk_len, out_dtype=torch.float32)
    dt = torch.rand(batch, seq_len, n_heads, dtype=torch.float32, device=run_device())
    A = -torch.rand(n_heads, dtype=torch.float32, device=run_device())

    test = SSDChunkCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    test.check(op, dt, A, None)


class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    pass


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
):
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )
    test.check(SSDChunkScanFwdOp(), *test.gen_inputs())


class SSDChunkStateFwdTest(SSDChunkStateFwdWorkload, TestBase):
    pass


@SSDChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx
):
    test = SSDChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx
    )
    op = SSDChunkStateFwdOp()
    if tune:
        op.request_tune()
    inputs = test.gen_inputs()
    test.check(op, *inputs)


@pytest.mark.smoke
def test_ssd_chunk_state_fwd_seq_idx_semantics():
    """Exercise negative chunk ends and the optional unmasked path."""
    batch, num_chunks, chunk_len = 1, 2, 64
    n_heads, d_head, d_state, n_groups = 4, 64, 32, 1
    dtype = torch.float16
    b, c, Q, h, p, n, g = batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups
    seq_len = c * Q

    x = torch.randn(b, seq_len, h, p, dtype=dtype, device=run_device()) * 0.1
    Bmat = torch.randn(b, seq_len, g, n, dtype=dtype, device=run_device()) * 0.1
    dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()).cumsum(-1)
    dt = torch.rand(b, h, c, Q, dtype=torch.float32, device=run_device()) * 0.1 + 0.01

    seq_idx = torch.ones(b, seq_len, dtype=torch.int32, device=run_device())
    seq_idx[:, :Q] = -1

    op = SSDChunkStateFwdOp()
    out = op(x, Bmat, dt, dA_cumsum, seq_idx)
    ref = ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, g, seq_idx=seq_idx)

    workload = SSDChunkStateFwdWorkload(b, c, Q, h, p, n, g, dtype, True)
    compare_outputs(out, ref, workload.verification(x))

    # A chunk whose sequence is -1 throughout is all zero.
    assert torch.count_nonzero(out[:, 0]) == 0
    assert out[:, 1].abs().max().item() > 0

    # A call without seq_idx must not read the freed seq_idx's memory.
    poison = torch.full((b, seq_len), -1, dtype=torch.int32, device=run_device())
    torch.accelerator.synchronize()
    del poison
    out = op(x, Bmat, dt, dA_cumsum)
    ref = ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, g)
    compare_outputs(out, ref, workload.verification(x))


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.sm90
def test_ssd_chunk_state_fp16_tma_masks_packed_sequences():
    from tileops.backend import BUILTIN

    test = SSDChunkStateFwdTest(1, 16, 256, 48, 64, 128, 1, torch.float16, True)
    x, bmat, dt, da, seq_idx = test.gen_inputs()
    seq_idx[:, :256] = -1
    seq_idx[:, 256:416] = 0
    seq_idx[:, 416:] = 1
    op = SSDChunkStateFwdOp(target=BUILTIN)
    inputs = (x, bmat, dt, da, seq_idx)
    out = op(*inputs)
    compare_outputs(out, test.ref_program(*inputs), test.verification(*inputs))
    (kernel,) = op.iter_kernels()
    assert kernel._uses_tma_panels()
    assert torch.count_nonzero(out[:, 0]) == 0


class SSDStatePassingFwdTest(SSDStatePassingFwdWorkload, TestBase):
    pass


@SSDStatePassingFwdFixture
def test_ssd_state_passing_fwd(batch, num_chunks, n_heads, d_state, dtype):
    test = SSDStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    test.check(SSDStatePassingFwdOp(), *test.gen_inputs())


class SSDDecodeTest(SSDDecodeWorkload, TestBase):
    pass


@SSDDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SSDDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SSDRecurrentFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs(), runs=lambda *args: ssd_decode_result(op, *args))


@pytest.mark.parametrize(
    "batch,seqlen,n_heads,d_head,d_state,n_groups,chunk_size,dtype",
    [
        pytest.param(1, 256, 4, 64, 32, 1, 256, torch.float16, marks=pytest.mark.smoke),
        pytest.param(1, 256, 4, 64, 32, 1, 256, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param(2, 512, 8, 64, 64, 2, 256, torch.float16, marks=pytest.mark.full),
        pytest.param(2, 512, 8, 64, 64, 2, 256, torch.bfloat16, marks=pytest.mark.full),
        pytest.param(1, 512, 4, 64, 128, 1, 256, torch.float16, marks=pytest.mark.full),
        pytest.param(1, 512, 4, 64, 128, 1, 256, torch.bfloat16, marks=pytest.mark.full),
    ],
)
def test_mamba2_fwd_e2e(batch, seqlen, n_heads, d_head, d_state, n_groups, chunk_size, dtype):
    """Mamba2FwdOp output must match the pure-PyTorch reference within tolerance."""
    dev = run_device()
    torch.manual_seed(42)
    x = torch.randn(batch, seqlen, n_heads, d_head, dtype=dtype, device=dev) * 0.1
    dt_raw = torch.randn(batch, seqlen, n_heads, dtype=torch.float32, device=dev) * 0.5
    A = -torch.rand(n_heads, dtype=torch.float32, device=dev)
    B = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    C = torch.randn(batch, seqlen, n_groups, d_state, dtype=dtype, device=dev) * 0.1
    dt_bias = torch.randn(n_heads, dtype=torch.float32, device=dev) * 0.1

    op = Mamba2FwdOp(chunk_size=chunk_size, dt_softplus=True)
    got = op(x, dt_raw, A, B, C, dt_bias=dt_bias)
    expected = mamba2_fwd_ref(x, dt_raw, A, B, C, dt_bias, chunk_size, dt_softplus=True)

    compare_outputs(got, expected, mamba2_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ssd_state_passing_fwd_vectorized_short_scan(dtype):
    """A short scan over a flattened 8192-wide state splits each thread's state in two."""
    test = SSDStatePassingFwdTest(2, 4, 8, 8192, dtype)
    test.check(SSDStatePassingFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ssd_state_passing_fwd_prefetched_scan(dtype):
    """A wide state over 16 chunks: four elements a thread, eight chunks loaded ahead."""
    test = SSDStatePassingFwdTest(1, 16, 4, 8192, dtype)
    test.check(SSDStatePassingFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_ssd_scan_retains_a_small_coefficient_on_a_large_value() -> None:
    test = SSDChunkScanFwdTest(1, 1, 64, 4, 64, 32, 1, torch.bfloat16)
    x, cb, decay, c, state, dt = test.gen_inputs()
    x.zero_()
    x[:, 0] = 1e8
    cb.zero_()
    cb[..., -1, 0] = 1
    decay.zero_()
    decay[..., 32:] = -20
    test.check(SSDChunkScanFwdOp(), x, cb, decay, c.zero_(), state.zero_(), dt.fill_(1))


class SSDStatePassingUnitTest(SSDStatePassingUnitWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "chunks,width,dtype,initial,period,tune",
    [
        pytest.param(
            32768, 32, torch.float32, True, 0, False, id="long-fp32", marks=pytest.mark.smoke
        ),
        pytest.param(
            32768, 32, torch.float16, True, 0, False, id="long-fp16", marks=pytest.mark.smoke
        ),
        pytest.param(
            32768, 32, torch.bfloat16, True, 0, False, id="long-bf16", marks=pytest.mark.smoke
        ),
        pytest.param(
            257, 128, torch.float32, False, 251, False, id="reset-tail", marks=pytest.mark.full
        ),
        pytest.param(
            264,
            4096,
            torch.bfloat16,
            True,
            251,
            False,
            id="wide-reset-tail",
            marks=pytest.mark.full,
        ),
        pytest.param(
            257, 128, torch.float16, True, 251, True, id="tuned-reset-tail", marks=pytest.mark.full
        ),
    ],
)
def test_ssd_state_passing_fwd_long_and_reset_scans(chunks, width, dtype, initial, period, tune):
    """Long scans and reset tails preserve previous and final states through the op."""
    test = SSDStatePassingUnitTest(1, chunks, 1, width, dtype, initial, reset_period=period)
    op = SSDStatePassingFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())
