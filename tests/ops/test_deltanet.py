"""Tests for the DeltaNet ops: chunkwise forward and backward, inference, decode."""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, served_in_tree
from tileops.backend import BUILTIN, TensorSpec, registry
from tileops.kernels.linear_attention import DeltaNetDensePrefillFwdKernel
from tileops.kernels.linear_attention.call_spec import DeltaNetChunkCall
from tileops.kernels.linear_attention.deltanet.chunk_bwd import DeltaNetBwdKernel
from tileops.kernels.linear_attention.deltanet.recurrent import (
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.linear_attention import (
    DeltaNetChunkBwdOp,
    DeltaNetChunkFwdOp,
)
from tileops.ops import DeltaNetInferenceFwdOp, DeltaNetRecurrentFwdOp
from workloads.device import run_device
from workloads.linear_attention.deltanet import (
    DeltaNetDecodeWorkload,
    DeltaNetFwdWorkload,
    DeltaNetInferenceWorkload,
    chunkwise_verification,
    decode_verification,
    deltanet_autograd_bwd_torch,
    deltanet_decode_torch,
)
from workloads.numerics import compare_outputs


class DeltaNetFwdTest(DeltaNetFwdWorkload, TestBase):
    pass


class DeltaNetFwdFixture(FixtureBase):
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
                pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
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
            ],
        ),
    ]


@DeltaNetFwdFixture
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = DeltaNetChunkFwdOp(chunk_size=chunk_size, tune=tune)
    test.check(op, *test.gen_inputs())
    if served_in_tree(op) and tune:
        # The forward above already proves the selected config builds and runs;
        # this pins it to the declared candidate set the sweep draws from.
        (kernel,) = op.built_kernels("deltanet_fwd").values()
        assert kernel.config in kernel.autotune_configs


class DeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
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


@DeltaNetBwdFixture
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

    # Forward to get S for backward kernel
    from tileops.ops import DeltaNetChunkFwdOp

    fwd_op = DeltaNetChunkFwdOp(chunk_size=BC)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op.forward(q, k, v, beta)
    do = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * 0.1

    # Reference via autograd
    ref_dq, ref_dk, ref_dv, ref_dbeta = deltanet_autograd_bwd_torch(do, q, k, v, beta, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dbeta)

    # Kernel
    op = DeltaNetChunkBwdOp(chunk_size=BC, tune=tune)
    op_outputs = op.forward(do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd)

    compare_outputs(
        op_outputs,
        tuple(t.to(dtype) for t in ref_outputs),
        chunkwise_verification(dtype, backward=True),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "budget, chunk_size, dim_k, dim_v, dtype, stages",
    [
        pytest.param(101376, 32, 64, 64, torch.float32, 1, id="sm89-c32-fp32"),
        pytest.param(101376, 64, 64, 64, torch.float16, 2, id="sm89-c64-fp16"),
        # Each refused by one program alone: the per-chunk backward, the recurrence, w/u.
        pytest.param(101376, 64, 64, 64, torch.float32, None, id="sm89-c64-fp32-refused"),
        pytest.param(101376, 32, 128, 128, torch.float16, None, id="sm89-c32-d128-refused"),
        pytest.param(101376, 64, 64, 128, torch.float16, None, id="sm89-dv128-refused"),
        # Refused by what the per-chunk backward holds once dP is written.
        pytest.param(166912, 64, 160, 16, torch.float32, None, id="sm80-dp-refused"),
        pytest.param(166912, 64, 128, 128, torch.float16, 1, id="sm80-d128"),
        pytest.param(232448, 64, 64, 64, torch.float16, 2, id="sm90-c64-fp16"),
        pytest.param(232448, 64, 128, 128, torch.float16, 1, id="sm90-d128"),
    ],
)
def test_deltanet_bwd_config_follows_the_shared_memory_budget(
    budget: int,
    chunk_size: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    stages: "int | None",
) -> None:
    """The recurrence's stage count and the refusal each follow their bound at a budget."""
    call = DeltaNetChunkCall(
        batch=1,
        heads=1,
        seq_len=4 * chunk_size,
        chunk_size=chunk_size,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype=dtype,
        arch=89,
        sm_count=1,
        smem_budget=budget,
    )
    if stages is None:
        assert "needs at least" in DeltaNetBwdKernel.refusal(call)
        return
    assert DeltaNetBwdKernel.refusal(call) is None
    config = DeltaNetBwdKernel._default_config_for(budget, chunk_size, dim_k, dim_v, dtype.itemsize)
    assert config["num_stages"] == stages


class DeltaNetInferenceTest(DeltaNetInferenceWorkload, TestBase):
    pass


@pytest.fixture
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
def test_deltanet_inference_reaches_target_with_optional_inputs() -> None:
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

    registry.register_kernel_builder("DeltaNetInferenceFwdOp", "deltanet_test", build_kernel)

    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    beta = torch.rand(1, 7, 2, dtype=torch.float16)
    initial_state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    cu_seqlens_cpu = cu_seqlens.clone()

    op = DeltaNetInferenceFwdOp(scale=0.125, use_qk_l2norm_in_kernel=True, target="deltanet_test")
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
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True},
        ),
        (
            tuple(
                None if tensor is None else TensorSpec.of(tensor)
                for tensor in (decode_q, decode_q, decode_v, decode_beta, decode_state, None, None)
            ),
            {"scale": 0.125, "use_qk_l2norm_in_kernel": True},
        ),
    ]


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_dense_prefill_matches_fla(dtype: torch.dtype) -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 128, 4, 64, dtype)
    inputs = test.gen_inputs()
    op = DeltaNetInferenceFwdOp()
    if dtype == torch.float16:
        test.check(op, *inputs)
        test.check(op, *inputs[:4])
    else:
        test.check(op, *inputs)
        test.check(op, *inputs[:4])


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_dense_prefill_normalizes_q_and_k() -> None:
    """``use_qk_l2norm_in_kernel`` takes Q and K unnormalized and matches FLA."""
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 128, 4, 64, torch.bfloat16, l2norm=True)
    op = DeltaNetInferenceFwdOp(use_qk_l2norm_in_kernel=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_prefill_packs_ragged_sequences() -> None:
    """Lengths below, across and on a chunk boundary in one packed call."""
    torch.manual_seed(42)
    test = DeltaNetInferenceTest(1, 0, 4, 64, torch.bfloat16, sequence_lengths=(1, 63, 100, 192))
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_prefill_runs_a_row_that_is_not_a_whole_chunk() -> None:
    torch.manual_seed(42)
    test = DeltaNetInferenceTest(2, 100, 4, 64, torch.bfloat16)
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_partitioned_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 512, 4, 64, torch.bfloat16)
    # 16 chunks split into partitions of 4.
    kernel = DeltaNetDensePrefillFwdKernel(
        2, 4, 512, 2, False, 64, 64**-0.5, torch.bfloat16, config={"max_local_chunks": 4}
    )
    inputs = [tensor.to("cuda") for tensor in test.gen_inputs()]
    test.check(DeltaNetInferenceFwdOp(), *inputs, runs=kernel)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_decode_matches_fla(dtype: torch.dtype) -> None:
    """One token continues a caller-owned state, and starts from zero without one."""
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 1, 4, 128, dtype)
    inputs = test.gen_inputs()
    op = DeltaNetInferenceFwdOp()
    # The workload preserves the single-step 4e-8 bound (measured error 4e-9).
    test.check(op, *inputs)
    test.check(op, *inputs[:4])


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_wide_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(1, 256, 4, 128, torch.bfloat16)
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs())


class DeltaNetDecodeTest(DeltaNetDecodeWorkload, TestBase):
    pass


class DeltaNetDecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            [
                pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@DeltaNetDecodeFixture
def test_deltanet_decode(
    batch: int, heads: int, dim_k: int, dim_v: int, dtype: torch.dtype, tune: bool
) -> None:
    torch.manual_seed(42)
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = DeltaNetRecurrentFwdOp(tune=tune)
    test.check(op, *test.gen_inputs())


@DeltaNetDecodeFixture
def test_deltanet_decode_multi_step(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test multiple sequential decode steps to verify state propagation."""
    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = batch, heads, dim_k, dim_v

    op = DeltaNetRecurrentFwdOp(tune=tune)

    state_op = torch.zeros(B, H, DK, DV, device=run_device(), dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device=run_device(), dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device=run_device(), dtype=dtype) * 0.1
        beta = torch.rand(B, H, device=run_device(), dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        compare_outputs(o_op, o_ref, decode_verification(dtype))
        compare_outputs(state_op, state_ref, decode_verification(dtype))


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_raw_cuda_real_128x128_smoke(dtype: torch.dtype) -> None:
    """PR smoke must compile and execute the real raw CUDA 128x128 fast path."""

    torch.manual_seed(42)
    test = DeltaNetDecodeTest(2, 4, 128, 128, dtype)
    op = DeltaNetRecurrentFwdOp(tune=False, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = op.built_kernels("deltanet_decode").values()
    assert isinstance(kernel, DeltaNetDecodeRawCudaFlaStyleKernel)
    test.check(op, *inputs)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_raw_cuda_real_128x128_multi_step_smoke(
    dtype: torch.dtype,
) -> None:
    """PR smoke must exercise raw CUDA state propagation across decode steps."""

    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = 2, 4, 128, 128
    op = DeltaNetRecurrentFwdOp(tune=False, target=BUILTIN)

    state_op = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device="cuda", dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=dtype) * 0.1
        beta = torch.rand(B, H, device="cuda", dtype=dtype) * 0.5

        o_ref, state_ref = deltanet_decode_torch(q, k, v, beta, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, beta, state_op)

        (kernel,) = op.built_kernels("deltanet_decode").values()
        assert isinstance(kernel, DeltaNetDecodeRawCudaFlaStyleKernel)

        compare_outputs(o_op, o_ref, decode_verification(dtype))
        compare_outputs(state_op, state_ref, decode_verification(dtype))


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.sm90
def test_deltanet_decode_raw_cuda_config_requires_full_warp_mapping() -> None:
    with pytest.raises(ValueError, match="threads .* must equal raw_group_size \\* v_tile"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 16,
                "v_tile": 16,
                "raw_group_size": 2,
                "raw_maxrregcount": 146,
            },
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.sm90
def test_deltanet_decode_raw_cuda_config_requires_two_lane_group() -> None:
    with pytest.raises(ValueError, match="raw_group_size must equal 2"):
        DeltaNetDecodeRawCudaFlaStyleKernel(
            1,
            32,
            128,
            128,
            dtype="bfloat16",
            config={
                "threads": 32,
                "v_tile": 8,
                "raw_group_size": 4,
                "raw_maxrregcount": 146,
            },
        )
