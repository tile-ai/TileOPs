"""Small shared primitives for MoE benchmark entry points."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def cuda_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / iters


def host_time_ms(fn: Callable[[], Any], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e3 / iters


def measure_dispatch(
    run_fresh: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
    make_cached: Callable[[Any], Callable[[], Any]] | None = None,
    validate_cached: Callable[[Any, Any], None] | None = None,
) -> tuple[Any, float, float | None]:
    """Measure fresh dispatch and an optional cached-handle path."""
    fresh_result = run_fresh()
    fresh_ms = cuda_time_ms(run_fresh, warmup=warmup, iters=iters)
    if make_cached is None:
        return fresh_result, fresh_ms, None
    run_cached = make_cached(fresh_result)
    cached_result = run_cached()
    cached_ms = cuda_time_ms(run_cached, warmup=warmup, iters=iters)
    if validate_cached is not None:
        validate_cached(fresh_result, cached_result)
    return fresh_result, fresh_ms, cached_ms


def make_routing_inputs(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    num_experts: int,
    *,
    distribution: str = "uniform",
    topk_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.rand(num_tokens, num_experts, device="cuda")
    if distribution == "uniform":
        scores = noise
    elif distribution == "hotspot":
        scores = torch.linspace(4.0, -4.0, num_experts, device="cuda").unsqueeze(0) + noise
    elif distribution == "longtail":
        ranks = torch.arange(1, num_experts + 1, device="cuda")
        scores = -torch.log(ranks).unsqueeze(0) + noise
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = scores.topk(top_k, dim=-1).indices.to(topk_dtype)
    weights = torch.softmax(torch.randn(num_tokens, top_k, device="cuda"), dim=-1)
    return hidden, topk_ids, weights


def make_expert_sizes(
    num_pairs: int,
    num_experts: int,
    distribution: str,
) -> list[int]:
    if distribution == "uniform":
        return [
            num_pairs // num_experts + (expert < num_pairs % num_experts)
            for expert in range(num_experts)
        ]
    generator = torch.Generator(device="cpu").manual_seed(42)
    if distribution == "longtail":
        probabilities = 1 / torch.arange(1, num_experts + 1, dtype=torch.float64).pow(1.2)
    elif distribution == "hotspot":
        hot_experts = max(1, num_experts // 8)
        probabilities = torch.full((num_experts,), 0.2 / num_experts)
        probabilities[:hot_experts] += 0.8 / hot_experts
    elif distribution == "router-like":
        probabilities = torch.softmax(torch.randn(num_experts, generator=generator) * 1.5, dim=0)
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    assignments = torch.multinomial(
        probabilities,
        num_samples=num_pairs,
        replacement=True,
        generator=generator,
    )
    return torch.bincount(assignments, minlength=num_experts).tolist()


def expert_mlp_reference(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    sizes: list[int],
) -> torch.Tensor:
    output = torch.empty(
        hidden.shape[0],
        hidden.shape[1],
        device=hidden.device,
        dtype=torch.float32,
    )
    ffn_size = w_down.shape[-1]
    start = 0
    for expert, size in enumerate(sizes):
        if size == 0:
            continue
        rows = hidden[start : start + size].float()
        gate_up = rows @ w_gate_up[expert].float().t()
        activated = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        output[start : start + size] = activated @ w_down[expert].float().t()
        start += size
    return output


def check_output(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_range_relative_tolerance: float,
    relative_l2_tolerance: float,
) -> tuple[float, float, float, float]:
    actual = actual.float()
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{name}: output contains NaN or Inf")
    error = actual - reference
    max_abs = error.abs().max().item()
    rmse = error.square().mean().sqrt().item()
    relative_l2 = (
        torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()
    max_range_relative = max_abs / reference.abs().max().clamp_min(1e-12).item()
    if max_range_relative > max_range_relative_tolerance or relative_l2 > relative_l2_tolerance:
        raise AssertionError(
            f"{name}: max_abs={max_abs}, rmse={rmse}, "
            f"relative_l2={relative_l2}, "
            f"max_range_relative={max_range_relative}"
        )
    return max_abs, rmse, relative_l2, max_range_relative


def effective_tflops(logical_flops: int, elapsed_ms: float) -> float:
    return logical_flops / (elapsed_ms / 1e3) / 1e12
