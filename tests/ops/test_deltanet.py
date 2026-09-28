"""Tests for the DeltaNet ops: chunkwise forward, backward and autograd, inference, decode."""

from functools import partial

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, allclose_compare, served_in_tree
from tileops.backend import BUILTIN, TensorSpec, registry
from tileops.kernels.linear_attention import DeltaNetDensePrefillFwdKernel
from tileops.kernels.linear_attention.deltanet_call import DeltaNetDecodeCall
from tileops.kernels.linear_attention.deltanet_recurrence import (
    DeltaNetDecodeFP32Kernel,
    DeltaNetDecodeKernel,
    DeltaNetDecodeRawCudaFlaStyleKernel,
)
from tileops.linear_attention import (
    DeltaNetAutogradFwdOp,
    DeltaNetBwdOp,
    DeltaNetFwdOp,
)
from tileops.ops import DeltaNetDecodeFwdOp, DeltaNetInferenceFwdOp
from workloads.device import run_device
from workloads.linear_attention import (
    DeltaNetDecodeWorkload,
    DeltaNetFwdWorkload,
    DeltaNetInferenceWorkload,
    deltanet_decode_torch,
)


class DeltaNetFwdTest(DeltaNetFwdWorkload, TestBase):
    pass


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 2e-2, "rtol": 2e-2}
    else:  # bfloat16
        return {"atol": 5e-2, "rtol": 5e-2}


class DeltaNetFwdFixture(FixtureBase):
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
    op = DeltaNetFwdOp(chunk_size=chunk_size, tune=tune)
    tols = _get_tolerances(dtype)
    inputs = test.gen_inputs()
    ref_o = test.ref_program(*inputs)
    op_o, _S, _Aw, _Au, _w, _u = op(*inputs)
    torch.testing.assert_close(op_o, ref_o, **tols)
    if served_in_tree(op) and tune:
        # The forward above already proves the selected config builds and runs;
        # this pins it to the declared candidate set the sweep draws from.
        (kernel,) = op.built_kernels("DeltaNetFwdKernel").values()
        assert kernel.config in kernel.autotune_configs


def _differentiable_fwd(q, k, v, beta, chunk_size):
    """Fully differentiable chunked forward matching DeltaNet (ungated)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        bc = beta[:, :, sl].float()
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        M = bc.unsqueeze(-1) * Gram
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        v_new = uc - wc @ h
        o_part = qc @ h
        attn = (qc @ kc.transpose(-2, -1)) * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        h = h + kc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def deltanet_autograd_bwd_torch(do, q, k, v, beta, chunk_size):
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = _differentiable_fwd(q_, k_, v_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dbeta = torch.autograd.grad(loss, [q_, k_, v_, beta_])
    return dq, dk, dv, dbeta


def _get_tolerances_deltanet_chunkwise_bwd(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 5e-3, "rtol": 5e-3}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


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
    from tileops.ops import DeltaNetFwdOp

    fwd_op = DeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, Aw, Au, w_fwd, u_fwd = fwd_op.forward(q, k, v, beta)
    do = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * 0.1

    # Reference via autograd
    ref_dq, ref_dk, ref_dv, ref_dbeta = deltanet_autograd_bwd_torch(do, q, k, v, beta, BC)
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dbeta)

    # Kernel
    op = DeltaNetBwdOp(chunk_size=BC, tune=tune)
    op_outputs = op.forward(do, q, k, v, beta, S_fwd, Aw, Au, w_fwd, u_fwd)

    tols = _get_tolerances_deltanet_chunkwise_bwd(dtype)
    names = ["dq", "dk", "dv", "dbeta"]
    for name, ref_out, op_out in zip(names, ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(
            op_out,
            ref_out.to(dtype),
            **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )


# The autograd wrapper owes only the wiring: its forward is the forward op's ``o``, and its
# backward produces what the backward op produces from the saved tensors. The numbers are
# checked by the forward and backward op tests.
B, H, S, DK, DV, BC = 1, 2, 256, 64, 64, 64


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(42)
    scale = 0.1
    q = torch.randn(B, H, S, DK, device=run_device(), dtype=dtype) * scale
    k = torch.randn(B, H, S, DK, device=run_device(), dtype=dtype) * scale
    v = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * scale
    beta = torch.rand(B, H, S, device=run_device(), dtype=dtype) * 0.5
    return q, k, v, beta


@pytest.mark.smoke
def test_deltanet_autograd_matches_the_ops_it_wraps() -> None:
    dtype = torch.float16
    q, k, v, beta = _inputs(dtype)
    do = torch.randn(B, H, S, DV, device=run_device(), dtype=dtype) * 0.1

    o_ref, s, aw, au, w, u = DeltaNetFwdOp(chunk_size=BC).forward(q, k, v, beta)
    grads_ref = DeltaNetBwdOp(chunk_size=BC).forward(do, q, k, v, beta, s, aw, au, w, u)

    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, beta)]
    o = DeltaNetAutogradFwdOp(chunk_size=BC)(*leaves)
    o.backward(do)

    torch.testing.assert_close(o, o_ref)
    for name, leaf, ref in zip(("dq", "dk", "dv", "dbeta"), leaves, grads_ref, strict=True):
        torch.testing.assert_close(leaf.grad, ref, msg=lambda m, n=name: f"{n}: {m}")


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
    assert op.eval_roofline() == (7 * 2 * (6 * 8 * 6 + 2 * 8), 2396)
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
        # The output meets the standard 1e-3 tolerance. The FP32 final state
        # has a measured 2.10e-3 maximum error for seeded and zero-state calls.
        compare = [
            partial(allclose_compare, atol=1e-3, rtol=1e-3),
            partial(allclose_compare, atol=3e-3, rtol=1e-3),
        ]
        test.check(op, *inputs, compare=compare)
        test.check(op, *inputs[:4], compare=compare)
    else:
        test.check(op, *inputs, atol=1.6e-2, rtol=1.6e-2)
        test.check(op, *inputs[:4], atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_partitioned_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(2, 512, 4, 64, torch.bfloat16)
    # 16 chunks split into partitions of 4.
    kernel = DeltaNetDensePrefillFwdKernel(
        2, 4, 512, 64, 64**-0.5, torch.bfloat16, config={"max_local_chunks": 4}
    )
    inputs = [tensor.to("cuda") for tensor in test.gen_inputs()]
    test.check(kernel, *inputs, atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.sm90
@pytest.mark.cuda_only
def test_deltanet_wide_prefill_matches_fla() -> None:
    torch.manual_seed(2163)
    test = DeltaNetInferenceTest(1, 256, 4, 128, torch.bfloat16)
    test.check(DeltaNetInferenceFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


class DeltaNetDecodeTest(DeltaNetDecodeWorkload, TestBase):
    pass


def _get_tolerances_deltanet_recurrence(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 5e-4, "rtol": 5e-4}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


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
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = DeltaNetDecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = DeltaNetDecodeFwdOp(tune=tune)
    tols = _get_tolerances_deltanet_recurrence(dtype)
    test.check(op, *test.gen_inputs(), **tols)


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

    op = DeltaNetDecodeFwdOp(tune=tune)
    tols = _get_tolerances_deltanet_recurrence(dtype)

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

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_deltanet_decode_raw_cuda_real_128x128_smoke(dtype: torch.dtype) -> None:
    """PR smoke must compile and execute the real raw CUDA 128x128 fast path."""

    torch.manual_seed(42)
    test = DeltaNetDecodeTest(2, 4, 128, 128, dtype)
    op = DeltaNetDecodeFwdOp(tune=False, target=BUILTIN)
    inputs = test.gen_inputs()
    op(*inputs)
    (kernel,) = op.built_kernels("deltanet_decode").values()
    assert isinstance(kernel, DeltaNetDecodeRawCudaFlaStyleKernel)
    test.check(op, *inputs, **_get_tolerances_deltanet_recurrence(dtype))


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
    op = DeltaNetDecodeFwdOp(tune=False, target=BUILTIN)
    tols = _get_tolerances_deltanet_recurrence(dtype)

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

        torch.testing.assert_close(o_op, o_ref, **tols)
        torch.testing.assert_close(state_op, state_ref, **tols)


class _DispatchMarker:
    """Records its construction instead of compiling anything.

    Mixed in ahead of the class each marker stands in for, so the region that
    class states — and the architecture it declares — still decide selection.
    Overriding only construction is the point: a marker that answered
    ``applies`` differently would be testing itself.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DefaultDispatchKernel(_DispatchMarker, DeltaNetDecodeKernel):
    pass


class _FP32DispatchKernel(_DispatchMarker, DeltaNetDecodeFP32Kernel):
    pass


class _RawDispatchKernel(_DispatchMarker, DeltaNetDecodeRawCudaFlaStyleKernel):
    pass


def _dispatch_kernel_map() -> dict:
    return {
        "DeltaNetDecodeKernel": _DefaultDispatchKernel,
        "DeltaNetDecodeFP32Kernel": _FP32DispatchKernel,
        "DeltaNetDecodeRawCudaFlaStyleKernel": _RawDispatchKernel,
    }


def _stated_call(
    sm_version: int, dtype: torch.dtype, dim_k: int = 128, dim_v: int = 128, tune: bool = False
) -> DeltaNetDecodeCall:
    """The record for a decode call, with the device stated rather than probed."""
    return DeltaNetDecodeCall(
        arch=sm_version, batch=1, heads=32, dim_k=dim_k, dim_v=dim_v, dtype=dtype, tune=tune
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_deltanet_decode_raw_cuda_dispatch_selects_raw_on_supported_sm90(
    dtype: torch.dtype,
) -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    assert op.select_kernel(_stated_call(90, dtype)) is _RawDispatchKernel


@pytest.mark.cuda_only
@pytest.mark.parametrize(
    "tune",
    [
        pytest.param(False, marks=pytest.mark.smoke, id="untuned"),
        pytest.param(True, marks=pytest.mark.full, id="tuned"),
    ],
)
def test_deltanet_decode_build_carries_the_tune_flag(tune: bool) -> None:
    """Whatever selection picks is constructed with the op's autotune setting."""
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map(), tune=tune)

    kernel = op.kernel_for("deltanet_decode", (), _stated_call(90, torch.bfloat16, tune=tune))

    assert kernel.kwargs["tune"] is tune


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_unsupported_sm() -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    assert op.select_kernel(_stated_call(80, torch.bfloat16)) is _DefaultDispatchKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(("dim_k", "dim_v"), [(64, 128), (128, 64)])
def test_deltanet_decode_raw_cuda_dispatch_falls_back_on_non_128_shapes(
    dim_k: int,
    dim_v: int,
) -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    assert (
        op.select_kernel(_stated_call(90, torch.bfloat16, dim_k, dim_v)) is _DefaultDispatchKernel
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_deltanet_decode_raw_cuda_dispatch_uses_fp32_kernel_for_fp32() -> None:
    op = DeltaNetDecodeFwdOp(kernel_map=_dispatch_kernel_map())

    assert op.select_kernel(_stated_call(90, torch.float32)) is _FP32DispatchKernel


@pytest.mark.cuda_only
@pytest.mark.smoke
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
