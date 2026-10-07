"""Benchmarks for the DeltaNet ops (chunkwise forward and backward, inference, decode), one case per
manifest call, against FLA and torch.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLA_TAG,
    TORCH_COMPILE_TAG,
    backward_of,
    compiled_reference,
    fla_op,
)
from tileops.ops import (
    DeltaNetChunkBwdOp,
    DeltaNetChunkFwdOp,
    DeltaNetInferenceFwdOp,
    DeltaNetRecurrentFwdOp,
)


# Chunkwise: FLA's chunk_delta_rule is required. TileOPs uses BHSD (q/k [B, H, S, DK], v [B, H, S, DV],
# beta [B, H, S]) and FLA BTHK, so inputs are permuted before calling FLA.
def _to_fla_layout(q, k, v, beta):
    """Convert TileOPs BHSD tensors to FLA BTHK layout."""
    return (
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )


@pytest.mark.parametrize("case", bench.cases(DeltaNetInferenceFwdOp), ids=lambda case: case.id)
def test_deltanet_inference_bench(case) -> None:
    op = DeltaNetInferenceFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"fla": case.reference})


@pytest.mark.parametrize("case", bench.cases(DeltaNetChunkFwdOp), ids=lambda case: case.id)
def test_deltanet_vs_fla_fwd(case) -> None:
    from fla.ops.delta_rule import chunk_delta_rule

    op = DeltaNetChunkFwdOp(**case.arguments)
    q_fla, k_fla, v_fla, beta_fla = _to_fla_layout(*case.inputs)

    def fla_fwd():
        out, state = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)
        return out.transpose(1, 2), state

    bench.Runner(op, case).compare({"fla": bench.Implementation(run=fla_fwd, args=())})


@pytest.mark.parametrize("case", bench.cases(DeltaNetChunkBwdOp), ids=lambda case: case.id)
def test_deltanet_vs_fla_bwd(case) -> None:
    from fla.ops.delta_rule import chunk_delta_rule

    do, q, k, v, beta, *_saved = case.inputs
    bwd_op = DeltaNetChunkBwdOp(**case.arguments)
    q_fla, k_fla, v_fla, beta_fla = (
        t.detach().requires_grad_(True) for t in _to_fla_layout(q, k, v, beta)
    )
    do_fla = do.permute(0, 2, 1, 3).contiguous()
    o_fla, _ = chunk_delta_rule(q_fla, k_fla, v_fla, beta_fla, scale=1.0)
    fla_backward = backward_of(o_fla)

    def fla_bwd():
        dq, dk, dv, dbeta = fla_backward(do_fla, None)[:4]
        return (dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2), dbeta.transpose(1, 2))

    bench.Runner(bwd_op, case).compare({"fla": bench.Implementation(run=fla_bwd, args=())})


@pytest.mark.parametrize("case", bench.cases(DeltaNetRecurrentFwdOp), ids=lambda case: case.id)
def test_deltanet_decode_bench(case) -> None:
    op = DeltaNetRecurrentFwdOp(**case.arguments)
    recurrent = fla_op("ops.delta_rule.fused_recurrent_delta_rule")

    def fla_fn(q, k, v, beta, state):
        out, final_state = recurrent(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            beta.unsqueeze(1),
            scale=1.0,
            initial_state=state.float(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        return out.squeeze(1), final_state.to(state.dtype)

    bench.Runner(op, case).compare(
        {
            FLA_TAG: fla_fn,
            "torch": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
