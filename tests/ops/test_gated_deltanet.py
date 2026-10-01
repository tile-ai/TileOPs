import pytest
import torch

from tests.test_base import TestBase
from tileops.backend import TensorSpec, registry
from tileops.kernels.linear_attention import GatedDeltaNetDensePrefillFwdKernel
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention import GatedDeltaNetFwdWorkload

pytestmark = pytest.mark.smoke


class GatedDeltaNetFwdTest(GatedDeltaNetFwdWorkload, TestBase):
    pass


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_gated_deltanet_dense_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 128, dtype)
    inputs = test.gen_inputs()
    op = GatedDeltaNetFwdOp()
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *inputs, atol=atol, rtol=rtol)


@pytest.mark.sm90
def test_gated_deltanet_dense_prefill_continues_an_initial_state() -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 128, torch.bfloat16, has_initial_state=True)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.sm90
def test_gated_deltanet_dense_prefill_runs_a_64_wide_state() -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 64, torch.bfloat16)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["from-zero", "continued"])
def test_gated_deltanet_partitioned_dense_prefill_matches_reference(
    has_initial_state: bool,
) -> None:
    """Exercise warmup, state correction, and partitioned forward together."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 512, 2, 128, torch.bfloat16, has_initial_state=has_initial_state)
    # 8 chunks split into partitions of 4.
    kernel = GatedDeltaNetDensePrefillFwdKernel(
        1, 2, 512, 128, 128**-0.5, torch.bfloat16, config={"max_local_chunks": 4}
    )
    q, k, v, g, beta, *state = (tensor.to("cuda") for tensor in test.gen_inputs())
    # A gentle decay, so the state carried across partitions still reaches the output.
    test.check(kernel, q, k, v, g * 0.01, beta, *state, atol=1.6e-2, rtol=1.6e-2)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_gated_deltanet_dense_decode_matches_reference(
    dtype: torch.dtype,
    batch: int,
) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, 1, 16, 128, dtype, has_initial_state=True)
    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.sm90
def test_gated_deltanet_dense_decode_propagates_fp32_state() -> None:
    torch.manual_seed(42)
    workload = GatedDeltaNetFwdWorkload(
        1,
        1,
        16,
        128,
        torch.bfloat16,
        has_initial_state=True,
    )
    q, k, v, g, beta, state = workload.gen_inputs()
    expected_state = state.clone()
    op = GatedDeltaNetFwdOp()
    for _ in range(4):
        expected_o, expected_state = workload.ref_program(q, k, v, g, beta, expected_state)
        got_o, state = op(q, k, v, g, beta, state)
        torch.testing.assert_close(got_o, expected_o, atol=1.6e-2, rtol=1.6e-2)
        torch.testing.assert_close(state, expected_state, atol=1.6e-2, rtol=1.6e-2)


def test_gated_deltanet_contract_reaches_target_builder() -> None:
    calls = []

    def build_kernel(*inputs, **params):
        calls.append((inputs, params))

        def kernel(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
        ):
            del k, g, beta, initial_state, cu_seqlens_cpu, A_log, dt_bias
            batch, seq_len, _heads, dim_k = q.shape
            value_heads, dim_v = v.shape[2:]
            state_batch = cu_seqlens.shape[0] - 1
            return (
                torch.empty(batch, seq_len, value_heads, dim_v, dtype=q.dtype),
                torch.empty(state_batch, value_heads, dim_k, dim_v, dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GatedDeltaNetFwdOp", "gdn_test", build_kernel)

    batch, seq_len, heads, value_heads, dim_k, dim_v = 1, 7, 2, 4, 8, 6
    q = torch.randn(batch, seq_len, heads, dim_k, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, value_heads, dim_v, dtype=torch.float16)
    g = torch.randn(batch, seq_len, value_heads, dtype=torch.float16)
    beta = torch.randn_like(g)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    cu_seqlens_cpu = cu_seqlens.clone()
    initial_state = torch.randn(2, value_heads, dim_k, dim_v, dtype=torch.float32)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, dtype=torch.float32)

    op = GatedDeltaNetFwdOp(
        scale=0.125,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        target="gdn_test",
    )
    o, final_state = op(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        A_log,
        dt_bias,
    )

    assert o.shape == (batch, seq_len, value_heads, dim_v)
    assert final_state.shape == (2, value_heads, dim_k, dim_v)
    assert calls == [
        (
            tuple(
                TensorSpec.of(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    cu_seqlens,
                    cu_seqlens_cpu,
                    A_log,
                    dt_bias,
                )
            ),
            {
                "scale": 0.125,
                "use_qk_l2norm_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "state_v_first": False,
                "use_gate_in_kernel": True,
            },
        )
    ]
