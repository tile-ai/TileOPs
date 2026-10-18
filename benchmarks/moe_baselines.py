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


def flashinfer_experts_skip_reason(hidden, w2, b_gate_up=None, b_down=None):
    """The reason ``cutlass_fused_moe``'s input checks refuse these experts, or ``None``."""
    # The binding reads both bias tensors when either is passed.
    if (b_gate_up is None) != (b_down is None):
        return "cutlass_fused_moe takes both expert biases or neither"
    # FlashInfer CUTLASS TMA rows require complete 16-byte units for both projections.
    elements = 16 // hidden.element_size()
    if hidden.shape[1] % elements or w2.shape[2] % elements:
        return f"cutlass_fused_moe needs H and F divisible by {elements}"
    return None


def flashinfer_experts(hidden, w1, w2, top_k, activation, b_gate_up=None):
    """Build a tuned CUTLASS adapter for fixed expert weights and token count.

    The activations and weight-row layouts follow ``MoEGroupedGemmFwdOp``. CUTLASS
    consumes the up rows before the gate rows. Repacked weights and biases retain
    their construction-time values and must remain unchanged. An untuned call profiles
    its exact token-count bucket; subsequent lookups use that same bucket. Profiling
    belongs to correctness warmup, outside measured calls.

    Args:
        hidden: Hidden states defining the token count, dtype and device.
        w1: Gate/up weights in the activation's row order.
        w2: Down weights defining the intermediate width.
        top_k: Routes per token.
        activation: The gated activation named by the benchmark's op.
        b_gate_up: Optional gate/up bias captured with ``w1``.

    Returns:
        A callable requiring the captured gate/up tensors, hidden/down signatures and routing shape.
    """
    run = flashinfer_op("fused_moe.cutlass_fused_moe")
    workspace_size = flashinfer_op("fused_moe.cutlass_fused_moe_workspace_size")
    activation_type = flashinfer_op("fused_moe.ActivationType")
    autotune = flashinfer_op("autotune", "autotuner")
    act = {
        "silu_and_mul": activation_type.Swiglu,
        "gelu_and_mul": activation_type.Geglu,
        "swigluoai": activation_type.SwigluBias,
    }[activation]
    tokens, experts = hidden.shape[0], w1.shape[0]
    hidden_signature = (tuple(hidden.shape), hidden.dtype, hidden.device)
    down_signature = (tuple(w2.shape), w2.dtype, w2.device)
    size = workspace_size(
        tokens,
        hidden.shape[1],
        w2.shape[-1],
        experts,
        top_k,
        x_dtype=hidden.dtype,
        weight_dtype=w1.dtype,
        output_dtype=hidden.dtype,
        activation_type=act,
        device=hidden.device,
    )
    workspace = torch.empty(size, dtype=torch.uint8, device=hidden.device)
    # w1 is stacked [gate; up], or gate/up interleaved for swigluoai.
    rows = torch.arange(w1.shape[1], device=w1.device)
    gate, up = (rows[0::2], rows[1::2]) if activation == "swigluoai" else rows.chunk(2)
    up_gate = torch.cat((up, gate))
    fc1 = w1.index_select(1, up_gate)
    bias_source = b_gate_up
    fc1_bias = None if b_gate_up is None else b_gate_up.index_select(1, up_gate)
    swiglu = {}
    if activation == "swigluoai":
        # swigluoai's fixed alpha, beta (the + 1 on up) and limit, as MoEGroupedGemmFwdOp's
        # docstring defines it.
        swiglu = {
            name: torch.full((experts,), value, dtype=torch.float32, device=hidden.device)
            for name, value in (
                ("swiglu_alpha", 1.702),
                ("swiglu_beta", 1.0),
                ("swiglu_limit", 7.0),
            )
        }
    # Profiling and lookup must use the same exact-token bucket.
    buckets = (tokens,)
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
            activation_type=act,
            workspace_buffer=workspace,
            **swiglu,
        )[0]

    def baseline(x, gate_up, down, weights, ids, b_gate_up=None, b_down=None):
        assert gate_up is w1 and b_gate_up is bias_source, "fc1 was repacked from other tensors"
        assert (tuple(x.shape), x.dtype, x.device) == hidden_signature, (
            "hidden states must match the captured shape, dtype and device"
        )
        assert (tuple(down.shape), down.dtype, down.device) == down_signature, (
            "down weights must match the captured shape, dtype and device"
        )
        assert tuple(weights.shape) == tuple(ids.shape) == (tokens, top_k), (
            "routing weights and ids must match the captured token and route counts"
        )
        if not tuned:
            with autotune(True, tuning_buckets=buckets):
                call(x, down, weights, ids, b_down)
            tuned.append(True)
        with autotune(False, tuning_buckets=buckets):
            return call(x, down, weights, ids, b_down)

    return baseline
