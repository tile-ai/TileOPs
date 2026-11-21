"""Benchmark Kimi Delta Attention (KDA), one case per manifest call, against FLA and, on a state
pool, vLLM."""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLA_TAG,
    VLLM_TAG,
    baseline_module,
    private_inputs,
    vllm_op,
    vllm_state_lines,
)
from tileops.linear_attention import KDAFwdOp


@pytest.mark.parametrize("case", bench.cases(KDAFwdOp), ids=lambda case: case.id)
def test_kda_fwd_bench(case) -> None:
    params = case.params
    inputs = case.inputs
    op = KDAFwdOp(**case.arguments)
    scale = inputs[0].shape[-1] ** -0.5 if params["scale"] is None else params["scale"]

    def fla_fn(q, k, v, g, beta, state, cu, cu_cpu, a_log, dt_bias, *_serving):
        fla_kda = baseline_module("fla.ops.kda", "fla")
        chunk_kda, fused_recurrent_kda = fla_kda.chunk_kda, fla_kda.fused_recurrent_kda

        # FLA caches derived metadata by tensor identity; snapshot the live call.
        cu = None if cu is None else cu.clone()
        cu_cpu = None if cu_cpu is None else cu_cpu.clone()
        arguments = dict(
            scale=params["scale"],
            use_gate_in_kernel=params["use_gate_in_kernel"],
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=params["lower_bound"],
            initial_state=state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=params["use_qk_l2norm_in_kernel"],
            use_beta_sigmoid_in_kernel=params["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=params["allow_neg_eigval"],
            state_v_first=params["state_v_first"],
            cu_seqlens=cu,
        )
        if q.shape[1] == 1:
            return fused_recurrent_kda(q, k, v, g, beta, **arguments)
        lower_bound = params["lower_bound"]
        # FLA applies the lower-bound gate either way; safe_gate only selects its faster
        # intra-chunk path, which FLA accepts for a lower bound in [-5, 0) alone.
        safe_gate = (
            params["use_gate_in_kernel"] and lower_bound is not None and -5 <= lower_bound < 0
        )
        return chunk_kda(q, k, v, g, beta, cu_seqlens_cpu=cu_cpu, safe_gate=safe_gate, **arguments)

    slots, accepted = inputs[10:12]
    if accepted is not None:
        # No baseline here keeps the state after every token of a sequence in a slot of its own.
        bench.Runner(op, case).compare({})
        return
    if slots is None:
        bench.Runner(op, case).compare({FLA_TAG: fla_fn})
        return

    # On a state pool each baseline runs on a private copy of the pool. FLA's recurrent kernel
    # reads and writes the slots of a decode step itself, through `fused_recurrent_kda_fwd`: its
    # public wrapper drops the slots, and over a longer sequence it stores the state after each
    # token to a slot of its own. A chunked prefill takes one state a sequence: the call gathers
    # its sequences' states from the pool and scatters their final states back, as vLLM's KDA
    # prefill does around its chunked kernel; both copies are timed.
    q, k, v, g, beta, pool, cu, *_ = inputs
    decode = q.shape[1] == 1
    if decode:
        recurrent_fla = baseline_module(
            "fla.ops.kda.fused_recurrent", "fla"
        ).fused_recurrent_kda_fwd

        def fla_pool(q, k, v, g, beta, pool, cu, cu_cpu, a_log, dt_bias, slots, _accepted):
            return recurrent_fla(
                q, k, v, g, beta, A_log=a_log, dt_bias=dt_bias, scale=scale, initial_state=pool,
                use_qk_l2norm_in_kernel=params["use_qk_l2norm_in_kernel"],
                use_gate_in_kernel=params["use_gate_in_kernel"],
                use_beta_sigmoid_in_kernel=params["use_beta_sigmoid_in_kernel"],
                allow_neg_eigval=params["allow_neg_eigval"], lower_bound=params["lower_bound"],
                state_v_first=params["state_v_first"], cu_seqlens=cu, ssm_state_indices=slots,
            )  # fmt: skip
    else:

        def fla_pool(q, k, v, g, beta, pool, cu, cu_cpu, a_log, dt_bias, slots, _accepted):
            o, final = fla_fn(q, k, v, g, beta, pool[slots], cu, cu_cpu, a_log, dt_bias)
            pool[slots] = final
            return o, pool

    implementations = {FLA_TAG: private_inputs(fla_pool, inputs, 5)}
    # vLLM keeps the state value-major, the layout of its pool, and reads the gate and the step
    # size transformed, as every pool row of the manifest has them.
    l2norm = params["use_qk_l2norm_in_kernel"]
    lines, line_of, reset_lines = vllm_state_lines(pool, slots)
    # vLLM's chunked kernel overwrites the values and the gate, so it runs on copies.
    values, decay = v.clone(), g.clone()

    def reset() -> None:
        reset_lines()
        values.copy_(v)
        decay.copy_(g)

    kda_ops = "third_party.flash_linear_attention.ops.kda"
    if decode:
        recurrent = vllm_op("fused_recurrent_kda", module=kda_ops)

        def run():
            o, _ = recurrent(
                q, k, v, g, beta, scale=scale, initial_state=lines,
                use_qk_l2norm_in_kernel=l2norm, ssm_state_indices=line_of,
            )  # fmt: skip
            return o, lines[1:]
    else:
        chunk = vllm_op("chunk_kda", module=kda_ops)

        def run():
            o, final = chunk(
                q, k, values, decay, beta, scale=scale, initial_state=lines[line_of],
                output_final_state=True, use_qk_l2norm_in_kernel=l2norm, cu_seqlens=cu,
            )  # fmt: skip
            lines[line_of] = final
            return o, lines[1:]

    implementations[VLLM_TAG] = bench.Implementation(run=run, args=(), reset=reset)

    # vLLM's Kimi KDA kernels take the raw gate and step-size logits and normalize Q and K: the
    # packed decode kernel reads the slots, and FlashKDA, its prefill, runs between a gather and
    # a scatter. The logits are formed here in float32 with A_log = 0 and dt_bias = 0: the step
    # size is sigmoid(b) for b = logit(beta); the decode kernel's log decay is -softplus(a) for
    # a = log(expm1(-g)), and FlashKDA's bounded decay lower * sigmoid(a) for a = logit(g / lower),
    # with lower below every g. Every pool row of the manifest normalizes Q and K over 128-wide
    # bfloat16 states with one value head a key head, which these kernels take.
    batch, _, heads, dim = q.shape
    lower = -5.0
    zeros = torch.zeros(heads * dim, device=q.device, dtype=torch.float32)
    rates = zeros[:heads]
    step = torch.logit(beta.float()).contiguous()
    if decode:
        packed = vllm_op(
            "fused_recurrent_kda_packed_decode", module="models.kimi_k3.nvidia.ops.third_party.kda"
        )
        mixed = torch.cat([x.reshape(batch, -1) for x in (q, k, v)], dim=1)
        raw = torch.log(torch.expm1(-g.float())).reshape(1, batch, heads, dim).contiguous()
        step = step.reshape(1, batch, heads)

        def kimi():
            o, _ = packed(
                mixed_qkv=mixed, raw_g=raw, raw_beta=step, A_log=rates, dt_bias=zeros,
                lower_bound=None, initial_state=lines, state_indices=line_of, scale=scale,
            )  # fmt: skip
            return o.reshape(v.shape), lines[1:]
    else:
        baseline_module("vllm._flashkda_C", "vllm")
        flash = torch.ops._flashkda_C
        # FlashKDA reads the gate and the step size in bfloat16.
        raw = torch.logit(g.float() / lower).to(torch.bfloat16).contiguous()
        step = step.to(torch.bfloat16)
        out = torch.empty_like(v)
        workspace = torch.empty(
            flash.get_workspace_size(batch * q.shape[1], heads, cu.numel() - 1),
            dtype=torch.uint8,
            device=q.device,
        )
        final = torch.empty((cu.numel() - 1, *pool.shape[1:]), dtype=pool.dtype, device=q.device)

        def kimi():
            flash.fwd(
                q, k, v, raw, step, scale, out, workspace, rates, zeros.view(heads, dim), lower,
                lines[line_of], final, cu,
            )  # fmt: skip
            lines[line_of] = final
            return out, lines[1:]

    implementations[f"{VLLM_TAG}-kimi"] = bench.Implementation(run=kimi, args=(), reset=reset)
    bench.Runner(op, case).compare(implementations)
