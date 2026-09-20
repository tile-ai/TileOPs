import pytest
import torch

from tests.test_base import TestBase
from tileops.backend import TensorSpec, registry
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the migrated dense-prefill specialization requires SM90",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_gated_deltanet_dense_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 128, dtype)
    inputs = test.gen_inputs()
    op = GatedDeltaNetFwdOp()
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float16 else (1.6e-2, 1.6e-2)
    test.check(op, *inputs, atol=atol, rtol=rtol)

    initial_state = torch.zeros(1, 2, 128, 128, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="initial_state"):
        op(*inputs, initial_state=initial_state)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the migrated dense-prefill specialization requires SM90",
)
def test_gated_deltanet_partitioned_dense_prefill_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise warmup, state correction, and partitioned forward together."""
    monkeypatch.setenv("TILEOPS_GDN_PREFILL_MAX_LOCAL_CHUNKS", "4")
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 512, 2, 128, torch.bfloat16)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs(), atol=1.6e-2, rtol=1.6e-2)


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


def test_gated_deltanet_rejects_invalid_optional_inputs() -> None:
    q = torch.empty(1, 1, 2, 8, dtype=torch.float16)
    k = torch.empty_like(q)
    v = torch.empty(1, 1, 2, 6, dtype=torch.float16)
    g = torch.empty(1, 1, 2, dtype=torch.float16)
    beta = torch.empty_like(g)

    with pytest.raises(ValueError, match="requires A_log and dt_bias"):
        GatedDeltaNetFwdOp(use_gate_in_kernel=True).forward(q, k, v, g, beta)

    bad_state = torch.empty(1, 2, 6, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"\[N, HV, K, V\]"):
        GatedDeltaNetFwdOp().forward(q, k, v, g, beta, bad_state)
