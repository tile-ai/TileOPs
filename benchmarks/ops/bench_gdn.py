"""Benchmark Gated DeltaNet (GDN) inference, one case per manifest call, against FLA and FlashInfer."""

import pytest
import torch
import torch.nn.functional as F

from benchmarks import api as bench
from benchmarks.baselines import FLASHINFER_TAG, flashinfer_op
from tileops.ops import GDNFwdOp


@pytest.mark.parametrize("case", bench.cases(GDNFwdOp), ids=lambda case: case.id)
def test_gdn_fwd_bench(case) -> None:
    ix = case.workload.call.ix
    inputs = case.inputs
    op = GDNFwdOp(**case.arguments)
    prefill = flashinfer_op("gdn_prefill.chunk_gated_delta_rule")

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        if ix["use_gate_in_kernel"]:
            g = -torch.exp(a_log) * F.softplus(g.float() + dt_bias)
        if ix["use_beta_sigmoid_in_kernel"]:
            beta = torch.sigmoid(beta.float()) * (2.0 if ix["allow_neg_eigval"] else 1.0)
        arguments = dict(
            scale=q.shape[-1] ** -0.5 if ix["scale"] is None else ix["scale"],
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=ix["use_qk_l2norm_in_kernel"],
            state_v_first=ix["state_v_first"],
        )
        if q.shape[1] == 1:
            return fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, **arguments)
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens_cpu=cu_cpu, **arguments)

    def flashinfer_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        offsets = cu
        if offsets is None:
            offsets = torch.arange(q.shape[0] + 1, dtype=torch.int64, device=q.device) * q.shape[1]
        if ix["use_gate_in_kernel"]:
            g = -torch.exp(a_log) * torch.nn.functional.softplus(g.float() + dt_bias)
        if ix["use_beta_sigmoid_in_kernel"]:
            factor = 2.0 if ix["allow_neg_eigval"] else 1.0
            beta = torch.sigmoid(beta.float()) * factor
        if state is not None and not ix["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
        dim = q.shape[-1]
        output_shape = v.shape
        if ix["use_qk_l2norm_in_kernel"]:
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
            scale=ix["scale"] if ix["scale"] is not None else dim**-0.5,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=False,
        )
        if not ix["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].reshape(output_shape), final[..., :dim, :dim].contiguous()

    decode = flashinfer_op("gated_delta_rule_decode_pretranspose", "gdn_decode")

    def decode_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias):
        if not ix["use_gate_in_kernel"]:
            a_log = torch.zeros(v.shape[2], device=q.device, dtype=torch.float32)
            dt_bias = torch.zeros_like(a_log)
            g = torch.log(torch.expm1(-g.float()))
        if not ix["use_beta_sigmoid_in_kernel"]:
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
        elif not ix["state_v_first"]:
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
            scale=ix["scale"] if ix["scale"] is not None else dim**-0.5,
            use_qk_l2norm=ix["use_qk_l2norm_in_kernel"],
        )
        if not ix["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].to(q.dtype), final[..., :dim, :dim].contiguous()

    functors = {"fla": fla_fn, FLASHINFER_TAG: flashinfer_fn}
    if inputs[0].shape[1] == 1:
        # A decode step is checked against the FP32 recurrence; both FlashInfer
        # paths carry BF16- or FP16-grade error into the FP32 state.
        functors[FLASHINFER_TAG] = bench.Implementation(
            run=flashinfer_fn,
            noncomparable_reason="chunk prefill kernel multiplies the FP32 state at input precision",
        )
        if inputs[6] is None and not ix["allow_neg_eigval"]:
            functors["flashinfer-decode"] = bench.Implementation(
                run=decode_fn,
                noncomparable_reason=(
                    "takes Q/K/V and the raw gate and step-size logits as BF16; converting "
                    "the workload's log decay and step size rounds both"
                ),
            )
    bench.Runner(op, case).compare(functors)
