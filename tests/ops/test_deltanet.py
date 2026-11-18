"""Tests for the DeltaNet ops: chunkwise forward and backward, inference, decode."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import DeltaNetChunkCall
from tileops.backend import BUILTIN, OpNotAvailableError, TensorSpec
from tileops.backend import _registry as registry
from tileops.linear_attention import (
    DeltaNetChunkBwdOp,
    DeltaNetChunkFwdOp,
    DeltaNetFwdOp,
    DeltaNetRecurrentFwdOp,
)
from workloads.device import run_device
from workloads.linear_attention.deltanet import (
    DeltaNetChunkwiseWorkload,
    DeltaNetDecodeWorkload,
    DeltaNetFwdWorkload,
    chunkwise_verification,
    decode_verification,
    deltanet_autograd_bwd_torch,
    deltanet_decode_torch,
)
from workloads.numerics import compare_outputs

# SM89 and SM90 prefill run different implementations. Each architecture's copy
# carries its marker, so ``-m sm89`` selects the SM89 copies.
PREFILL_ARCHITECTURES = pytest.mark.parametrize(
    "_architecture",
    [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)],
)


class DeltaNetChunkwiseTest(DeltaNetChunkwiseWorkload, TestBase):
    pass


class DeltaNetChunkFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(
                    2,
                    64,
                    2,
                    64,
                    64,
                    32,
                    torch.float32,
                    False,
                    marks=[pytest.mark.smoke, pytest.mark.packaging(family="linear_attention")],
                ),
                pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
                # dim_k 128 at chunk 128: SM89's 99 KB of shared memory per block holds the
                # recurrence but not the state walk.
                pytest.param(
                    1,
                    256,
                    4,
                    128,
                    128,
                    128,
                    torch.bfloat16,
                    False,
                    marks=(pytest.mark.smoke, pytest.mark.sm89),
                ),
                *[
                    pytest.param(
                        1,
                        seq_len,
                        2,
                        dim_k,
                        dim_v,
                        chunk,
                        dtype,
                        False,
                        marks=pytest.mark.full,
                        id=name,
                    )
                    for name, seq_len, dim_k, dim_v, chunk, dtype in (
                        ("chunk16-fp16", 64, 64, 64, 16, torch.float16),
                        # An odd count of 8-column tiles: no column split above one warp divides it.
                        ("chunk16-v24-fp32", 64, 64, 24, 16, torch.float32),
                        ("chunk16-v24-fp16", 64, 64, 24, 16, torch.float16),
                        ("chunk16-v48", 64, 64, 48, 16, torch.float16),
                        ("chunk16-walk", 64, 16, 64, 16, torch.bfloat16),
                        # 2 by 3 warp tiles, which TileLang's square split divides only at one warp.
                        ("chunk32-v24", 64, 64, 24, 32, torch.bfloat16),
                        # One 16-row tile in the walk operands' products.
                        ("chunk64-k16", 128, 16, 64, 64, torch.float16),
                    )
                ],
                pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                # State walk with dim_v > dim_k: the late-load branches.
                # This float32 forward at chunk 128, head dims 64 / 128, fits only SM90's
                # shared memory.
                pytest.param(
                    1,
                    256,
                    2,
                    64,
                    128,
                    128,
                    torch.float32,
                    False,
                    marks=(pytest.mark.full, pytest.mark.sm90),
                ),
                # chunk_size=64 is where the untuned default takes a tiled width, so
                # the tuned run has to beat a tiled baseline rather than no tiling.
                pytest.param(
                    2,
                    128,
                    2,
                    64,
                    64,
                    64,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned",
                ),
                *[
                    pytest.param(
                        1, 64, 2, dim_k, dim_v, 16, dtype, True, marks=pytest.mark.nightly, id=name
                    )
                    for name, dim_k, dim_v, dtype in (
                        ("chunk16-tuned", 64, 64, torch.bfloat16),
                        ("chunk16-tuned-v24", 64, 24, torch.float16),
                        ("chunk16-tuned-v96", 64, 96, torch.float16),
                        ("chunk16-walk-tuned", 16, 64, torch.bfloat16),
                    )
                ],
            ],
        ),
    ]


@DeltaNetChunkFwdFixture
def test_deltanet_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = DeltaNetChunkwiseTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = DeltaNetChunkFwdOp(chunk_size=chunk_size)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_deltanet_fwd_serves_chunk_128_in_sm89_shared_memory() -> None:
    """Dispatch on SM89's device facts serves dim_k 128 at chunk 128: where the state walk
    does not fit a block's shared memory, the recurrence does."""
    (in_tree,) = Kernel.devices
    call = DeltaNetChunkCall(
        arch=89,
        sm_count=142,
        smem_budget=101376,
        batch=1,
        heads=4,
        seq_len=256,
        chunk_size=128,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
        device=torch.device(in_tree),
    )
    op = DeltaNetChunkFwdOp(chunk_size=128)
    assert op.key_for("deltanet_chunk_fwd", call) == "deltanet_chunk_fwd"


class DeltaNetChunkBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
                # SM89's 99 KB of shared memory per block holds the w/u program at head dims
                # 128 / 128 only with its storage reused.
                pytest.param(
                    1,
                    256,
                    4,
                    128,
                    128,
                    64,
                    torch.bfloat16,
                    False,
                    marks=(pytest.mark.smoke, pytest.mark.sm89),
                ),
                # dim_k 16 gives the per-chunk gradients and the dh walk one 16-row warp tile.
                pytest.param(
                    1,
                    128,
                    2,
                    16,
                    64,
                    64,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                    id="chunk64-k16",
                ),
                pytest.param(
                    1,
                    256,
                    16,
                    128,
                    128,
                    64,
                    torch.float16,
                    False,
                    marks=pytest.mark.full,
                ),
                pytest.param(
                    1,
                    256,
                    16,
                    128,
                    128,
                    64,
                    torch.bfloat16,
                    False,
                    marks=pytest.mark.full,
                ),
                # 128 heads fill a wave with the walk's widest value tile, which at head dims
                # 128 / 128 SM89's shared memory cannot hold.
                pytest.param(
                    8,
                    256,
                    16,
                    128,
                    128,
                    64,
                    torch.bfloat16,
                    False,
                    marks=(pytest.mark.full, pytest.mark.sm89),
                ),
                # SM89's 99 KB of shared memory per block holds the w/u program's float32 bound
                # at head dims 64 / 64 only with its storage reused.
                pytest.param(
                    1,
                    256,
                    4,
                    64,
                    64,
                    64,
                    torch.float32,
                    False,
                    marks=(pytest.mark.full, pytest.mark.sm89),
                ),
                # SM90's shared memory holds the w/u program at head dims 256 / 256 only with
                # its storage reused.
                pytest.param(
                    1,
                    256,
                    4,
                    256,
                    256,
                    32,
                    torch.bfloat16,
                    False,
                    marks=(pytest.mark.full, pytest.mark.sm90),
                ),
                # Backward from the state walk's saved S, w and u.
                pytest.param(1, 256, 2, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(
                    2,
                    64,
                    2,
                    64,
                    64,
                    32,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned",
                ),
            ],
        ),
    ]


@DeltaNetChunkBwdFixture
def test_deltanet_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device=run_device(), dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device=run_device(), dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * 0.1
    beta = torch.rand(B, H, S, device=run_device(), dtype=dtype) * 0.5

    fwd_op = DeltaNetChunkFwdOp(chunk_size=BC)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op(q, k, v, beta)
    do = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * 0.1

    ref_dq, ref_dk, ref_dv, ref_dbeta = deltanet_autograd_bwd_torch(do, q, k, v, beta, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dbeta)

    op = DeltaNetChunkBwdOp(chunk_size=BC)
    if tune:
        op.request_tune()
    op_outputs = op(do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd)

    compare_outputs(
        op_outputs,
        tuple(t.to(dtype) for t in ref_outputs),
        chunkwise_verification(dtype, backward=True),
    )


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "chunk_size, dim_k, dim_v, dtype",
    [(64, 64, 64, torch.float32), (64, 128, 128, torch.bfloat16)],
)
def test_deltanet_bwd_serves_in_sm89_shared_memory(
    chunk_size: int, dim_k: int, dim_v: int, dtype: torch.dtype
) -> None:
    """Dispatch on SM89's device facts serves these calls: with the w/u program's storage
    reused, its shared memory as written fits a block's."""
    (in_tree,) = Kernel.devices
    call = DeltaNetChunkCall(
        arch=89,
        sm_count=142,
        smem_budget=101376,
        batch=1,
        heads=4,
        seq_len=256,
        chunk_size=chunk_size,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype=dtype,
        device=torch.device(in_tree),
    )
    assert DeltaNetChunkBwdOp().key_for("deltanet_chunk_bwd", call) == "deltanet_chunk_bwd"


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "chunk_size, dim_k, dim_v, dtype",
    [
        (64, 256, 256, torch.bfloat16),
        (128, 32, 256, torch.bfloat16),
        (64, 32, 256, torch.float32),
    ],
)
def test_deltanet_bwd_refused_beyond_sm90_shared_memory(
    chunk_size: int, dim_k: int, dim_v: int, dtype: torch.dtype
) -> None:
    """Dispatch on SM90's device facts refuses these calls: where the gradient program of each
    chunk forms the partial du it still holds the chunk's attention tile, and what it holds
    there exceeds a block's shared memory."""
    (in_tree,) = Kernel.devices
    call = DeltaNetChunkCall(
        arch=90,
        sm_count=132,
        smem_budget=232448,
        batch=1,
        heads=4,
        seq_len=256,
        chunk_size=chunk_size,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype=dtype,
        device=torch.device(in_tree),
    )
    with pytest.raises(OpNotAvailableError, match="needs at least"):
        DeltaNetChunkBwdOp().key_for("deltanet_chunk_bwd", call)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "op_cls, name, chunk_size, dim_k, dim_v",
    [
        (DeltaNetChunkFwdOp, "deltanet_chunk_fwd", 40, 64, 64),
        (DeltaNetChunkFwdOp, "deltanet_chunk_fwd", 64, 24, 64),
        (DeltaNetChunkBwdOp, "deltanet_chunk_bwd", 64, 64, 24),
    ],
)
def test_deltanet_chunk_refuses_extents_no_matmul_tiles(
    op_cls, name, chunk_size, dim_k, dim_v
) -> None:
    """No launch builds a chunk_size or dim_k off the 16-row matmul tile, nor a backward that
    reduces a 16-bit dim_v off the 16-wide matmul step, so dispatch refuses them."""
    (in_tree,) = Kernel.devices
    call = DeltaNetChunkCall(
        arch=90,
        sm_count=132,
        smem_budget=232448,
        batch=1,
        heads=2,
        seq_len=128,
        chunk_size=chunk_size,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype=torch.bfloat16,
        device=torch.device(in_tree),
    )
    with pytest.raises(OpNotAvailableError, match="multiple of 16"):
        op_cls().key_for(name, call)


class DeltaNetFwdTest(DeltaNetFwdWorkload, TestBase):
    pass


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
def test_deltanet_fwd_reaches_target_with_optional_inputs() -> None:
    calls = []

    def build_kernel(*specs, **params):
        calls.append((specs, params))

        def kernel(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu):
            del k, beta, initial_state, cu_seqlens_cpu
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return (
                torch.empty_like(v),
                torch.empty(state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("DeltaNetFwdOp", "deltanet_test", build_kernel)

    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    beta = torch.rand(1, 7, 2, dtype=torch.float16)
    initial_state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    cu_seqlens_cpu = cu_seqlens.clone()

    op = DeltaNetFwdOp(scale=0.125, use_qk_l2norm_in_kernel=True, target="deltanet_test")
    o, final_state = op(q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)

    assert o.shape == v.shape
    assert final_state.shape == initial_state.shape
    assert op.eval_roofline() == (7 * 2 * (6 * 8 * 6 + 2 * 8) + 2 * 7 * 2 * (3 * 8 + 2), 2396)
    decode_q = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    decode_v = torch.randn(1, 1, 2, 6, dtype=torch.float16)
    decode_beta = torch.rand(1, 1, 2, dtype=torch.float16)
    decode_state = torch.zeros(1, 2, 8, 6, dtype=torch.float32)
    decode_o, decode_final_state = op(decode_q, decode_q, decode_v, decode_beta, decode_state)
    assert decode_o.shape == decode_v.shape
    assert decode_final_state.shape == decode_state.shape
    assert calls == [
        (
            tuple(
                TensorSpec.of(tensor)
                for tensor in (q, k, v, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
            ),
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True, "validate_inputs": False},
        ),
        (
            tuple(
                None if tensor is None else TensorSpec.of(tensor)
                for tensor in (decode_q, decode_q, decode_v, decode_beta, decode_state, None, None)
            ),
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True, "validate_inputs": False},
        ),
    ]


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@PREFILL_ARCHITECTURES
def test_deltanet_dense_prefill_matches_fla(_architecture: int, dtype: torch.dtype) -> None:
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(2, 128, 4, 64, dtype)
    inputs = test.gen_inputs()
    op = DeltaNetFwdOp()
    test.check(op, *inputs)
    test.check(op, *inputs[:4])


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_deltanet_dense_prefill_normalizes_q_and_k(_architecture: int) -> None:
    """``use_qk_l2norm_in_kernel`` takes Q and K unnormalized and matches FLA."""
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(2, 128, 4, 64, torch.bfloat16, l2norm=True)
    op = DeltaNetFwdOp(use_qk_l2norm_in_kernel=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_deltanet_prefill_packs_ragged_sequences(_architecture: int) -> None:
    """Lengths below, across and on a chunk boundary in one packed call."""
    torch.manual_seed(42)
    test = DeltaNetFwdTest(1, 0, 4, 64, torch.bfloat16, sequence_lengths=(1, 63, 100, 192))
    test.check(DeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_deltanet_prefill_runs_a_row_that_is_not_a_whole_chunk(_architecture: int) -> None:
    torch.manual_seed(42)
    test = DeltaNetFwdTest(2, 100, 4, 64, torch.bfloat16)
    test.check(DeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.sm90
def test_deltanet_partitioned_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(2, 512, 4, 64, torch.bfloat16)
    inputs = [tensor.to("cuda") for tensor in test.gen_inputs()]
    # 16 chunks fall to the four-chunk partition floor, so sequences cross partitions.
    test.check(DeltaNetFwdOp(target=BUILTIN), *inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.cuda_only("cuda-reference")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_decode_matches_fla(dtype: torch.dtype) -> None:
    """One token continues a caller-owned state, and starts from zero without one."""
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(2, 1, 4, 128, dtype)
    inputs = test.gen_inputs()
    op = DeltaNetFwdOp()
    # The workload keeps the single-step 4e-8 bound.
    test.check(op, *inputs)
    test.check(op, *inputs[:4])


@pytest.mark.smoke
@pytest.mark.sm90
def test_deltanet_wide_partitioned_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(2, 1024, 4, 128, torch.bfloat16)
    # 16 chunks a row over a four-chunk partition: a 128-wide state crosses partitions.
    test.check(DeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@PREFILL_ARCHITECTURES
def test_deltanet_wide_prefill_matches_fla(_architecture: int) -> None:
    torch.manual_seed(2163)
    test = DeltaNetFwdTest(1, 256, 4, 128, torch.bfloat16)
    test.check(DeltaNetFwdOp(), *test.gen_inputs())


class DeltaNetDecodeTest(DeltaNetDecodeWorkload, TestBase):
    pass


class DeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype",
            [
                pytest.param(1, 4, 64, 64, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 128, 128, torch.float32, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.bfloat16, marks=pytest.mark.full),
                # A key dim the default key tile covers in one tile, whose state load hung
                # on SM90 when lowered to a TMA bulk copy.
                pytest.param(1, 4, 16, 64, torch.float32, marks=pytest.mark.full),
                pytest.param(1, 4, 16, 64, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@DeltaNetDecodeFixture
def test_deltanet_decode(
    batch: int, heads: int, dim_k: int, dim_v: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(42)
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    test.check(DeltaNetRecurrentFwdOp(), *test.gen_inputs())


def _check_decode_steps(op, batch: int, heads: int, dim_k: int, dim_v: int, dtype) -> None:
    """Eight decode steps from a zero state, each output and state checked against the
    reference."""
    torch.manual_seed(42)
    device = run_device()
    state_op = torch.zeros(batch, heads, dim_k, dim_v, device=device, dtype=dtype)
    state_ref = torch.zeros(batch, heads, dim_k, dim_v, device=device, dtype=dtype)
    for _ in range(8):
        q = torch.randn(batch, heads, dim_k, device=device, dtype=dtype) * 0.1
        k = torch.randn(batch, heads, dim_k, device=device, dtype=dtype) * 0.1
        v = torch.randn(batch, heads, dim_v, device=device, dtype=dtype) * 0.1
        beta = torch.rand(batch, heads, device=device, dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        compare_outputs(o_op, o_ref, decode_verification(dtype))
        compare_outputs(state_op, state_ref, decode_verification(dtype))


@DeltaNetDecodeFixture
def test_deltanet_decode_multi_step(
    batch: int, heads: int, dim_k: int, dim_v: int, dtype: torch.dtype
) -> None:
    _check_decode_steps(DeltaNetRecurrentFwdOp(), batch, heads, dim_k, dim_v, dtype)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_warp_real_128x128_smoke(dtype: torch.dtype) -> None:
    """PR smoke must compile and execute the real raw CUDA 128x128 fast path."""

    torch.manual_seed(42)
    test = DeltaNetDecodeTest(2, 4, 128, 128, dtype)
    op = DeltaNetRecurrentFwdOp(target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_warp_real_128x128_multi_step_smoke(dtype: torch.dtype) -> None:
    """PR smoke must exercise raw CUDA state propagation across decode steps."""
    _check_decode_steps(DeltaNetRecurrentFwdOp(target=BUILTIN), 2, 4, 128, 128, dtype)
