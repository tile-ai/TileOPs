from typing import Any

import torch

from workloads.workload_base import WorkloadBase


class FusedTopKWorkload(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, scoring_func, renormalize, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        torch.manual_seed(42)
        return (torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device="cuda"),)


class MoePermuteWorkload(WorkloadBase):

    def __init__(self, total_tokens, top_k, num_experts, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = torch.randn(
            self.total_tokens, self.hidden_size, dtype=self.dtype, device="cuda"
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return hidden_states, topk_ids


class MoePermuteAlignWorkload(WorkloadBase):

    def __init__(self, total_tokens: int, top_k: int, num_experts: int, block_size: int):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.block_size = block_size

    def gen_inputs(self) -> tuple[torch.Tensor]:
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.total_tokens, self.top_k),
            dtype=torch.int32, device="cuda",
        )
        return (topk_ids,)


class MoeUnpermuteWorkload(WorkloadBase):

    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = self.total_tokens * self.top_k
        mm2_pad = torch.randn(numel, self.hidden_size, dtype=self.dtype, device="cuda")
        # fwd_idx: simulate a valid mapping: random shuffle of [0, numel)
        fwd_idx = torch.randperm(numel, dtype=torch.int32, device="cuda")
        topk_weights = torch.rand(
            self.total_tokens, self.top_k, dtype=torch.float32, device="cuda"
        )
        return mm2_pad, fwd_idx, topk_weights


def make_expert_sizes_offsets(
    numel: int,
    num_experts: int,
    distribution: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (true_sizes, true_offsets) for a fixed token-to-expert distribution.

    Args:
        numel: Total token-expert pairs (T * top_k).
        num_experts: Number of experts E.
        distribution: "uniform" — evenly split; "skewed" — most tokens on first
            20% of experts (one-token floor for the rest).
        device: CUDA device string.

    Returns:
        true_sizes [E] int32, true_offsets [E] int32.
    """
    if distribution == "uniform":
        base = max(1, numel // num_experts)
        sizes = torch.full((num_experts,), base, dtype=torch.int32, device=device)
        sizes[-1] = numel - base * (num_experts - 1)
    elif distribution == "skewed":
        sizes = torch.ones(num_experts, dtype=torch.int32, device=device)
        extra = numel - num_experts
        top_experts = max(1, num_experts // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    assert int(sizes.sum().item()) == numel
    return sizes, offsets


class MoeGroupedGemmNopadWorkload(WorkloadBase):
    """Tight A, per-expert weights B, and the expert size/offset tables."""

    def __init__(
        self,
        numel: int,
        num_experts: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        distribution: str = "uniform",
    ):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.dtype = dtype
        self.distribution = distribution

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        true_sizes, true_offsets = make_expert_sizes_offsets(
            self.numel, self.num_experts, self.distribution, dev
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        return a, b, true_sizes, true_offsets


class FusedMoeWorkload(WorkloadBase):
    """Inputs for a single FusedMoe benchmark configuration."""

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_size: int,
        scoring_func: str,
        renormalize: bool,
        with_correction_bias: bool,
        routed_scaling_factor: float,
        dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev,
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=torch.float32, device=dev,
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
            if self.with_correction_bias else None
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        return hidden, gating, correction_bias, w_gate_up, w_down


class SharedFusedMoeWorkload(WorkloadBase):
    def __init__(
        self,
        num_tokens,
        num_experts,
        top_k,
        hidden_size,
        ffn_size,
        shared_ffn_size,
        scoring_func,
        renormalize,
        with_correction_bias,
        routed_scaling_factor,
        dtype,
    ):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.shared_ffn_size = shared_ffn_size
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev
        )
        gating = torch.randn(
            self.num_tokens, self.num_experts, dtype=self.dtype, device=dev
        )
        correction_bias = (
            torch.randn(self.num_experts, dtype=torch.float32, device=dev) * 0.1
            if self.with_correction_bias else None
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        # Shared expert weights: gate+up concatenated [2*Fs, H], down [H, Fs]
        shared_w_gate_up = torch.randn(
            self.shared_ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev
        ) * 0.02
        shared_w_down = torch.randn(
            self.hidden_size, self.shared_ffn_size, dtype=self.dtype, device=dev
        ) * 0.02
        return hidden, gating, correction_bias, w_gate_up, w_down, shared_w_gate_up, shared_w_down


class MoeExpertsWorkload(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        w1 = torch.randn(self.num_experts, self.ffn_size * 2, self.hidden_size, dtype=self.dtype, device=dev) * 0.02
        w2 = torch.randn(self.num_experts, self.hidden_size, self.ffn_size, dtype=self.dtype, device=dev) * 0.02
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev), dim=-1
        )
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=torch.int32, device=dev)
        return hidden, w1, w2, topk_weights, topk_ids


class MoeFusedActivationWorkload(WorkloadBase):
    """Workload descriptor for fused vs unfused activation benchmark."""

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype,
    ):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        # Primary shape: (num_tokens, hidden_size) — the token tensor footprint.
        self.shape: tuple[int, int] = (num_tokens, hidden_size)

    def gen_inputs(self) -> tuple[Any, ...]:
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev,
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev),
            dim=-1,
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.num_tokens, self.top_k), dtype=torch.int32, device=dev,
        )
        return hidden, w_gate_up, w_down, topk_weights, topk_ids


class MoeSharedExpertMlpWorkload(WorkloadBase):
    def __init__(self, num_tokens, hidden_size, ffn_size, dtype):
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self):
        device = torch.device("cuda")
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        w_gate_up = torch.randn(self.ffn_size * 2, self.hidden_size, dtype=self.dtype, device=device)
        w_down = torch.randn(self.hidden_size, self.ffn_size, dtype=self.dtype, device=device)
        return hidden, w_gate_up, w_down
