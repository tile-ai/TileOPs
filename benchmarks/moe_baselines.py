"""Library expert MLPs with caller-owned scratch space."""

import torch

from benchmarks.baselines import flashinfer_op, vllm_op


def vllm_experts(activation="silu_and_mul"):
    """vLLM Triton ``fused_experts`` on *activation*, with the per-expert biases a case passes.

    ``silu_and_mul``, ``gelu_and_mul`` and ``swigluoai`` map to ``MoEActivation``'s
    ``SILU``, ``GELU`` (the erf form) and ``SWIGLUOAI``, which reads ``w1`` gate/up
    interleaved as the case stores it. The biases ride in ``biased_moe_quant_config``;
    vLLM adds ``b_down`` before the routing weight.
    """
    fused_experts = vllm_op("fused_experts", "model_executor.layers.fused_moe.fused_moe")
    moe_activation = vllm_op("MoEActivation", "model_executor.layers.fused_moe.activation")
    biased_config = vllm_op("biased_moe_quant_config", "model_executor.layers.fused_moe.config")
    act = {
        "silu_and_mul": moe_activation.SILU,
        "gelu_and_mul": moe_activation.GELU,
        "swigluoai": moe_activation.SWIGLUOAI,
    }[activation]

    def baseline(x, gate_up, down, weights, ids, b_gate_up=None, b_down=None):
        quant_config = None
        if b_gate_up is not None or b_down is not None:
            quant_config = biased_config(b_gate_up, b_down)
        return fused_experts(
            x, gate_up, down, weights, ids, activation=act, quant_config=quant_config
        )

    return baseline


def _interleaved_to_up_gate(t):
    """Rows 2j gate_j / 2j + 1 up_j along dim 1 to the stacked [up; gate] CUTLASS reads."""
    return torch.cat((t[:, 1::2], t[:, 0::2]), dim=1).contiguous()


def _stacked_to_up_gate(t):
    """The gate half, then the up half, along dim 1 to the [up; gate] CUTLASS reads."""
    gate, up = t.chunk(2, dim=1)
    return torch.cat((up, gate), dim=1).contiguous()


def flashinfer_experts(hidden, w1, w2, top_k, activation="silu_and_mul", b_gate_up=None):
    """CUTLASS permute, gated expert GEMMs and weighted route reduction.

    ``silu_and_mul`` and ``gelu_and_mul`` run ``Swiglu`` and ``Geglu`` (the erf form), which
    read fc1 and its bias stacked [up; gate]; ``b_gate_up`` is repacked here, outside the
    timer, so a call must pass the tensor the adapter was built from.
    """
    if activation == "swigluoai":
        return _flashinfer_swigluoai(hidden, w1, w2, top_k, b_gate_up)
    run = flashinfer_op("fused_moe.cutlass_fused_moe")
    workspace_size = flashinfer_op("fused_moe.cutlass_fused_moe_workspace_size")
    activation_type = flashinfer_op("fused_moe.ActivationType")
    act = {"silu_and_mul": activation_type.Swiglu, "gelu_and_mul": activation_type.Geglu}[
        activation
    ]
    size = workspace_size(
        hidden.shape[0],
        hidden.shape[1],
        w2.shape[-1],
        w1.shape[0],
        top_k,
        x_dtype=hidden.dtype,
        weight_dtype=w1.dtype,
        output_dtype=hidden.dtype,
        activation_type=act,
        device=hidden.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device=hidden.device)
    bias_source = b_gate_up
    fc1_bias = None if b_gate_up is None else _stacked_to_up_gate(b_gate_up)

    def baseline(x, gate_up, down, weights, ids, b_gate_up=None, b_down=None):
        assert b_gate_up is bias_source, "fc1's bias was repacked from another tensor"
        # CUTLASS SwiGLU stores the linear half before the activated half.
        gate, up = gate_up.chunk(2, dim=1)
        up_gate = torch.cat((up, gate), dim=1)
        return run(
            x,
            ids.int(),
            weights.float(),
            up_gate,
            down,
            x.dtype,
            [],
            fc1_expert_biases=fc1_bias,
            fc2_expert_biases=b_down,
            activation_type=act,
            workspace_buffer=workspace,
        )[0]

    return baseline


def _flashinfer_swigluoai(hidden, w1, w2, top_k, b_gate_up):
    """gpt-oss experts on the CUTLASS ``SwigluBias`` path, weights repacked once, tactics tuned.

    Passing ``swiglu_alpha/beta/limit`` selects ``SwigluBias``: g * sigmoid(alpha * g) *
    (u + beta) on min(gate, limit) and clamp(up, -limit, limit), biases added in fp32 first.
    It reads fc1 stacked [up; gate], so the interleaved ``w1`` and ``b_gate_up`` are repacked
    here, outside the timer; a call must pass the tensors the adapter was built from.
    The first call profiles the CUTLASS tactics for this token count (``autotune``), so the
    timed calls run the tuned tactic.
    """
    run = flashinfer_op("fused_moe.cutlass_fused_moe")
    workspace_size = flashinfer_op("fused_moe.cutlass_fused_moe_workspace_size")
    activation_type = flashinfer_op("fused_moe.ActivationType")
    autotune = flashinfer_op("autotune", "autotuner")
    experts = w1.shape[0]
    size = workspace_size(
        hidden.shape[0],
        hidden.shape[1],
        w2.shape[-1],
        experts,
        top_k,
        x_dtype=hidden.dtype,
        weight_dtype=w1.dtype,
        output_dtype=hidden.dtype,
        activation_type=activation_type.SwigluBias,
        device=hidden.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device=hidden.device)
    fc1 = _interleaved_to_up_gate(w1)
    bias_source = b_gate_up
    fc1_bias = None if b_gate_up is None else _interleaved_to_up_gate(b_gate_up)

    def per_expert(value):
        return torch.full((experts,), value, dtype=torch.float32, device=hidden.device)

    # swigluoai's fixed alpha, beta (the + 1 on up) and limit, as MoEGroupedGemmFwdOp's
    # docstring defines it.
    swiglu = {
        "swiglu_alpha": per_expert(1.702),
        "swiglu_beta": per_expert(1.0),
        "swiglu_limit": per_expert(7.0),
    }
    tuned = []

    def call(x, down, weights, ids, b_down):
        return run(
            x,
            ids.int(),
            weights.float(),
            fc1,
            down,
            x.dtype,
            [],
            fc1_expert_biases=fc1_bias,
            fc2_expert_biases=b_down,
            workspace_buffer=workspace,
            **swiglu,
        )[0]

    def baseline(x, gate_up, down, weights, ids, b_gate_up=None, b_down=None):
        assert gate_up is w1 and b_gate_up is bias_source, "fc1 was repacked from other tensors"
        if not tuned:
            with autotune(True, tuning_buckets=(x.shape[0],)):
                call(x, down, weights, ids, b_down)
            tuned.append(True)
        return call(x, down, weights, ids, b_down)

    return baseline
