"""Library expert MLPs with caller-owned scratch space."""

import torch

from benchmarks.baselines import flashinfer_op
from benchmarks.verification import Custom, NegativeControl, assert_normalized_error


def flashinfer_experts(hidden, w1, w2, top_k):
    """CUTLASS permute, gated expert GEMMs and weighted route reduction."""
    run = flashinfer_op("fused_moe.cutlass_fused_moe")
    workspace_size = flashinfer_op("fused_moe.cutlass_fused_moe_workspace_size")
    size = workspace_size(
        hidden.shape[0],
        hidden.shape[1],
        w2.shape[-1],
        w1.shape[0],
        top_k,
        x_dtype=hidden.dtype,
        weight_dtype=w1.dtype,
        output_dtype=hidden.dtype,
        device=hidden.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device=hidden.device)

    def baseline(x, gate_up, down, weights, ids):
        # CUTLASS SwiGLU stores the linear half before the activated half.
        gate, up = gate_up.chunk(2, dim=1)
        up_gate = torch.cat((up, gate), dim=1)
        return run(
            x, ids.int(), weights.float(), up_gate, down, x.dtype, [], workspace_buffer=workspace
        )[0]

    return baseline


def moe_evidence(gate_up_index):
    """Check both individual errors and scale-independent energy, including gate order."""

    def validate(got, expected):
        if isinstance(expected, (tuple, list)):
            for value, target in zip(got, expected, strict=True):
                validate(value, target)
        elif expected is None:
            assert got is None
        else:
            torch.testing.assert_close(got, expected, rtol=3e-2, atol=3e-2)
            assert_normalized_error(got, expected, tolerance=1e-4)

    def swapped(reference, inputs):
        changed = list(inputs)
        gate, up = changed[gate_up_index].chunk(2, dim=-2)
        changed[gate_up_index] = torch.cat((up, gate), dim=-2)
        return reference(*changed)

    return Custom(
        validate,
        "elementwise atol/rtol 3e-2 and normalized squared error <= 1e-4",
        controls=(NegativeControl("gate-up-swapped", swapped),),
    )
