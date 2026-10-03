"""Library expert MLPs with caller-owned scratch space."""

import torch

from benchmarks.baselines import flashinfer_op


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
        return run(
            x, ids.int(), weights.float(), gate_up, down, x.dtype, [], workspace_buffer=workspace
        )[0]

    return baseline
