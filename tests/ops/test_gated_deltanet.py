import pytest
import torch

from tileops.backend import TensorSpec, registry
from tileops.ops import GatedDeltaNetFwdOp

pytestmark = pytest.mark.smoke


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
