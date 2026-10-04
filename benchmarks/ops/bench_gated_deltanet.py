"""Benchmark Gated DeltaNet inference, one case per manifest call, against FLA and FlashInfer."""

import pytest
import torch
import torch.nn.functional as F

from benchmarks.baselines import FLASHINFER_TAG, flashinfer_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention.gated_deltanet import GatedDeltaNetFwdCall


@pytest.mark.parametrize("call", manifest_calls(GatedDeltaNetFwdOp))
def test_gated_deltanet_fwd_bench(call) -> None:
    workload = GatedDeltaNetFwdCall(call)
    inputs = workload.gen_inputs()
    op = GatedDeltaNetFwdOp(**workload.arguments())
    prefill = flashinfer_op("gdn_prefill.chunk_gated_delta_rule")
    offsets = inputs[6]
    if offsets is None:
        offsets = (
            torch.arange(inputs[0].shape[0] + 1, dtype=torch.int64, device=inputs[0].device)
            * inputs[0].shape[1]
        )

    def flashinfer_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        if call.ix["use_gate_in_kernel"]:
            g = (-torch.exp(a_log) * torch.nn.functional.softplus(g.float() + dt_bias)).to(g.dtype)
        if call.ix["use_beta_sigmoid_in_kernel"]:
            factor = 2.0 if call.ix["allow_neg_eigval"] else 1.0
            beta = (torch.sigmoid(beta.float()) * factor).to(beta.dtype)
        if state is not None and not call.ix["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
        dim = q.shape[-1]
        output_shape = v.shape
        if call.ix["use_qk_l2norm_in_kernel"]:
            # FlashInfer's prefill API accepts the flag but does not apply normalization.
            q = (q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(
                q.dtype
            )
            k = (k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(
                k.dtype
            )
        if dim < 128:
            q, k, v = (F.pad(x, (0, 128 - dim)) for x in (q, k, v))
            if state is not None:
                state = F.pad(state, (0, 128 - dim, 0, 128 - dim))
        out, final = prefill(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            g=g.float().exp().flatten(0, 1),
            beta=beta.float().flatten(0, 1),
            scale=call.ix["scale"] if call.ix["scale"] is not None else dim**-0.5,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=False,
        )
        if not call.ix["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].reshape(output_shape), final[..., :dim, :dim].contiguous()

    decode = flashinfer_op("gated_delta_rule_decode_pretranspose", "gdn_decode")

    def decode_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        if not call.ix["use_gate_in_kernel"]:
            a_log = torch.zeros(v.shape[2], device=q.device, dtype=torch.float32)
            dt_bias = torch.zeros_like(a_log)
            g = torch.log(torch.expm1(-g.float()))
        if not call.ix["use_beta_sigmoid_in_kernel"]:
            beta = torch.logit(beta.float())
        if state is None:
            state = torch.zeros(
                q.shape[0],
                v.shape[2],
                v.shape[-1],
                k.shape[-1],
                device=q.device,
                dtype=torch.float32,
            )
        elif not call.ix["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
        else:
            state = state.clone()
        dim = q.shape[-1]
        if dim < 128:
            q, k, v = (F.pad(x, (0, 128 - dim)) for x in (q, k, v))
            state = F.pad(state, (0, 128 - dim, 0, 128 - dim))
        # The installed decode kernel loads Q/K/V as BF16, including its FP16 API path.
        out, final = decode(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            state,
            a_log,
            g.to(torch.bfloat16),
            dt_bias,
            beta.to(torch.bfloat16),
            scale=call.ix["scale"] if call.ix["scale"] is not None else dim**-0.5,
            use_qk_l2norm=call.ix["use_qk_l2norm_in_kernel"],
        )
        if not call.ix["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].to(q.dtype), final[..., :dim, :dim].contiguous()

    functors = {"tileops": op, "fla": workload.ref_program, FLASHINFER_TAG: flashinfer_fn}
    if inputs[0].shape[1] == 1 and inputs[6] is None and not call.ix["allow_neg_eigval"]:
        functors["flashinfer-decode"] = decode_fn
    ManifestBenchmark(op, workload).compare(
        functors,
        *inputs,
    )
