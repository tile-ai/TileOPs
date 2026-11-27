"""Benchmark Gated DeltaNet (GDN) inference, one case per manifest call, against FLA, FlashInfer
and, on a state pool, vLLM."""

import pytest
import torch
import torch.nn.functional as F

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    flashinfer_op,
    private_inputs,
    vllm_op,
    vllm_state_lines,
)
from tileops.linear_attention import GDNFwdOp


def _raw_logits(g: torch.Tensor, beta: torch.Tensor):
    """The raw gate and step-size logits whose in-kernel transforms give *g* and *beta*, formed
    in float32 so neither rounds: the log decay g is -softplus(a) for a = log(expm1(-g)) with
    A_log = 0 and dt_bias = 0, the step size sigmoid(b) for b = logit(beta)."""
    return torch.log(torch.expm1(-g.float())), torch.logit(beta.float())


def _draft_verification(case) -> dict:
    """The implementations that verify drafts on a state pool: each starts a sequence from the
    slot of its last accepted token and keeps the state after each draft in a slot of its own.

    vLLM's recurrent kernel is the one the manifest names; its Qwen MTP layer runs the same
    recurrence with the gate and step size formed in kernel, and FlashInfer's MTP kernel takes
    them the same way, as raw logits formed outside the timed call. The draft-verification rows
    keep the pool value-major, the layout all three take, hand the gate and step size over
    transformed, and pack sequences of one length, the form FlashInfer takes.
    """
    params = case.params
    q, k, v, g, beta, pool, cu, _cu_cpu, _a_log, _dt_bias, slots, accepted = case.inputs
    implementations = {"torch-ref": case.torch}
    if cu is None:
        # Packed offsets drive the reference's loop over tokens, which compiles to no one graph.
        implementations[TORCH_COMPILE_TAG] = compiled_reference(case.torch)
    scale = q.shape[-1] ** -0.5 if params["scale"] is None else params["scale"]
    l2norm = params["use_qk_l2norm_in_kernel"]
    lines, line_of, reset_lines = vllm_state_lines(pool, slots)
    raw_gate, step = _raw_logits(g, beta)
    rates = biases = torch.zeros(v.shape[2], device=q.device, dtype=torch.float32)

    recurrent = vllm_op(
        "fused_recurrent_gated_delta_rule", module="third_party.flash_linear_attention.ops"
    )
    gated = vllm_op(
        "fused_sigmoid_gating_delta_rule_update",
        module="third_party.flash_linear_attention.ops.fused_sigmoid_gating",
    )

    def vllm_recurrent():
        o, _ = recurrent(
            q, k, v, g, beta, scale=scale, initial_state=lines, inplace_final_state=True,
            cu_seqlens=cu, ssm_state_indices=line_of, num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=l2norm,
        )  # fmt: skip
        return o, lines[1:]

    def vllm_gated():
        o, _ = gated(
            A_log=rates, a=raw_gate, b=step, dt_bias=biases, q=q, k=k, v=v, scale=scale,
            initial_state=lines, inplace_final_state=True, cu_seqlens=cu,
            ssm_state_indices=line_of, num_accepted_tokens=accepted,
            use_qk_l2norm_in_kernel=l2norm,
        )  # fmt: skip
        return o, lines[1:]

    implementations[VLLM_TAG] = bench.Implementation(run=vllm_recurrent, args=(), reset=reset_lines)
    implementations["vllm-mtp"] = bench.Implementation(run=vllm_gated, args=(), reset=reset_lines)

    rows = slots.shape[0]
    tokens = q.shape[0] * q.shape[1] // rows
    mtp = flashinfer_op("gated_delta_rule_mtp", "gdn_decode")
    states = pool.clone()
    # The slot each sequence starts from is formed outside the timed call.
    first = slots.gather(1, accepted.long()[:, None] - 1)[:, 0].contiguous()
    shaped = [x.reshape(rows, tokens, *x.shape[2:]) for x in (q, k, v)]
    gate, logit = (x.reshape(rows, tokens, x.shape[-1]).contiguous() for x in (raw_gate, step))
    token_slots = slots[:, :tokens].contiguous()
    out = torch.empty_like(shaped[2])

    def reset_states() -> None:
        states.copy_(pool)

    def flashinfer_mtp():
        mtp(
            *shaped, states, first, rates, gate, biases, logit, scale=scale, output=out,
            ssm_state_indices=token_slots, disable_state_update=False, use_qk_l2norm=l2norm,
        )  # fmt: skip
        return out.reshape(v.shape), states

    implementations[FLASHINFER_TAG] = bench.Implementation(
        run=flashinfer_mtp, args=(), reset=reset_states
    )
    return implementations


@pytest.mark.parametrize("case", bench.cases(GDNFwdOp))
def test_gdn_fwd_bench(case) -> None:
    params = case.params
    inputs = case.inputs
    op = GDNFwdOp(**case.arguments)
    prefill = flashinfer_op("gdn_prefill.chunk_gated_delta_rule")

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias, *_serving):
        fla_rule = baseline_module("fla.ops.gated_delta_rule", "fla")
        chunk_gated_delta_rule = fla_rule.chunk_gated_delta_rule
        fused_recurrent_gated_delta_rule = fla_rule.fused_recurrent_gated_delta_rule

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        if params["use_gate_in_kernel"]:
            g = -torch.exp(a_log) * F.softplus(g.float() + dt_bias)
        if params["use_beta_sigmoid_in_kernel"]:
            beta = torch.sigmoid(beta.float()) * (2.0 if params["allow_neg_eigval"] else 1.0)
        arguments = dict(
            scale=q.shape[-1] ** -0.5 if params["scale"] is None else params["scale"],
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=params["use_qk_l2norm_in_kernel"],
            state_v_first=params["state_v_first"],
        )
        if q.shape[1] == 1:
            return fused_recurrent_gated_delta_rule(q, k, v, g=g, beta=beta, **arguments)
        return chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens_cpu=cu_cpu, **arguments)

    def flashinfer_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias, *_serving):
        offsets = cu
        if offsets is None:
            offsets = torch.arange(q.shape[0] + 1, dtype=torch.int64, device=q.device) * q.shape[1]
        if params["use_gate_in_kernel"]:
            g = -torch.exp(a_log) * torch.nn.functional.softplus(g.float() + dt_bias)
        if params["use_beta_sigmoid_in_kernel"]:
            factor = 2.0 if params["allow_neg_eigval"] else 1.0
            beta = torch.sigmoid(beta.float()) * factor
        if state is not None and not params["state_v_first"]:
            state = state.transpose(-1, -2).contiguous()
        dim = q.shape[-1]
        output_shape = v.shape
        if params["use_qk_l2norm_in_kernel"]:
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
            scale=params["scale"] if params["scale"] is not None else dim**-0.5,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=offsets,
            use_qk_l2norm_in_kernel=False,
        )
        if not params["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].reshape(output_shape), final[..., :dim, :dim].contiguous()

    decode = flashinfer_op("gated_delta_rule_decode_pretranspose", "gdn_decode")

    def decode_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias, *_serving):
        if not params["use_gate_in_kernel"]:
            a_log = torch.zeros(v.shape[2], device=q.device, dtype=torch.float32)
            dt_bias = torch.zeros_like(a_log)
            g = torch.log(torch.expm1(-g.float()))
        if not params["use_beta_sigmoid_in_kernel"]:
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
        elif not params["state_v_first"]:
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
            scale=params["scale"] if params["scale"] is not None else dim**-0.5,
            use_qk_l2norm=params["use_qk_l2norm_in_kernel"],
        )
        if not params["state_v_first"]:
            final = final.transpose(-1, -2).contiguous()
        return out[..., :dim].to(q.dtype), final[..., :dim, :dim].contiguous()

    implementations = {"fla": fla_fn, FLASHINFER_TAG: flashinfer_fn}
    if inputs[0].shape[1] == 1:
        # A decode step is checked against the FP32 recurrence; both FlashInfer
        # paths carry BF16- or FP16-grade error into the FP32 state.
        implementations[FLASHINFER_TAG] = bench.Implementation(
            run=flashinfer_fn,
            noncomparable_reason="chunk prefill kernel multiplies the FP32 state at input precision",
        )
        if inputs[6] is None and not params["allow_neg_eigval"]:
            implementations["flashinfer-decode"] = bench.Implementation(
                run=decode_fn,
                noncomparable_reason=(
                    "takes Q/K/V and the raw gate and step-size logits as BF16; converting "
                    "the workload's log decay and step size rounds both"
                ),
            )
    slots, accepted = inputs[10:12]
    if accepted is not None:
        implementations = _draft_verification(case)
    elif slots is not None:
        # Each baseline takes one state a sequence: the call gathers its sequences' states from
        # a private pool and scatters their final states back, as vLLM's GDN prefill does
        # around its chunked kernel; both copies are timed.
        def gathered(implementation):
            if not isinstance(implementation, bench.Implementation):
                implementation = bench.Implementation(run=implementation)
            run = implementation.run

            def call(q, k, v, g, beta, pool, cu, cu_cpu, a_log, dt_bias, slots, _accepted):
                o, final = run(q, k, v, g, beta, pool[slots], cu, cu_cpu, a_log, dt_bias)
                pool[slots] = final
                return o, pool

            return private_inputs(
                call, inputs, 5, noncomparable_reason=implementation.noncomparable_reason
            )

        implementations = {tag: gathered(impl) for tag, impl in implementations.items()}
        # vLLM keeps the state value-major, the layout of its pool; the pool rows hand the gate
        # and the step size over transformed.
        transformed = not (params["use_gate_in_kernel"] or params["use_beta_sigmoid_in_kernel"])
        if params["state_v_first"] and transformed:
            # vLLM's packed decode kernel, which its Qwen layers run for a decode step, reads and
            # writes the slots; its chunked prefill runs between a gather and a scatter. The packed
            # kernel reads Q, K and V as one row a token and the gate and step size as raw logits
            # formed outside the timed call.
            q, k, v, g, beta, pool, cu = inputs[:7]
            lines, line_of, reset = vllm_state_lines(pool, slots)
            scale = q.shape[-1] ** -0.5 if params["scale"] is None else params["scale"]
            l2norm = params["use_qk_l2norm_in_kernel"]

            if q.shape[1] == 1:
                packed = vllm_op(
                    "fused_recurrent_gated_delta_rule_packed_decode",
                    module="third_party.flash_linear_attention.ops.fused_recurrent",
                )
                batch, value_heads = v.shape[0], v.shape[2]
                mixed = torch.cat([x.reshape(batch, -1) for x in (q, k, v)], dim=1)
                raw_gate, step = (x.reshape(batch, value_heads) for x in _raw_logits(g, beta))
                rates = biases = torch.zeros(value_heads, device=q.device, dtype=torch.float32)
                step = step.contiguous()
                out = torch.empty_like(v)

                def run():
                    packed(
                        mixed_qkv=mixed, a=raw_gate.contiguous(), b=step, A_log=rates,
                        dt_bias=biases, scale=scale, initial_state=lines, out=out,
                        ssm_state_indices=line_of, use_qk_l2norm_in_kernel=l2norm,
                    )  # fmt: skip
                    return out, lines[1:]

                vllm = bench.Implementation(run=run, args=(), reset=reset)
            else:
                chunk = vllm_op(
                    "chunk_gated_delta_rule", module="third_party.flash_linear_attention.ops.chunk"
                )

                def run():
                    o, final = chunk(
                        q, k, v, g, beta, scale=scale, initial_state=lines[line_of],
                        output_final_state=True, cu_seqlens=cu, use_qk_l2norm_in_kernel=l2norm,
                    )  # fmt: skip
                    lines[line_of] = final
                    return o, lines[1:]

                vllm = bench.Implementation(run=run, args=(), reset=reset)
            implementations[VLLM_TAG] = vllm
    bench.Runner(op, case).compare(implementations)
