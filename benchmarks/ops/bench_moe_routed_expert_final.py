"""Benchmark: TileOPs MoE Routed Expert Performance Report

Comparison implementations:
1. TileOPs (Separate): gate_up GEMM → activation → down GEMM
2. TileOPs (Fused): gate_up GEMM with fused activation → down GEMM
3. CUTLASS Grouped GEMM: reference data (GEMM only)

Run:
    conda activate tileops-tl9
    python benchmarks/ops/bench_moe_routed_expert_final.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from tilelang.profiler import do_bench

from tileops.ops.moe.routed_expert.fused_routed_expert import FusedMoEExpertsNopadPersistent3WGFwdOp

# CUTLASS reference numbers (H200, SM90, CUDA 12.9)
# These are grouped GEMM performance numbers, excluding routing/activation
# Used as theoretical upper bound reference
_CUTLASS_GROUPED_GEMM_TFLOPS = {
    # (M_per_expert, N, K, num_experts): TFLOPS (Cooperative schedule)
    (64, 256, 1024, 64): 429,   # corresponds to 512 tokens, 64 experts, top_k=8
    (512, 256, 1024, 64): 429,  # corresponds to 4096 tokens, 64 experts, top_k=8
    (32, 256, 2048, 128): 493,  # corresponds to 512 tokens, 128 experts, top_k=8
    (256, 256, 2048, 128): 493, # corresponds to 4096 tokens, 128 experts, top_k=8
}


def _bench(fn, warmup=10, rep=100) -> float:
    """Benchmark a function and return latency in ms."""
    with torch.no_grad():
        ms = do_bench(fn, warmup=warmup, rep=rep, backend='cupti')
        if ms <= 0:
            ms = do_bench(fn, warmup=warmup, rep=rep,
                          backend='event', return_mode='median')
    return ms


def _build_uniform_routing(num_tokens, num_experts, top_k, device):
    """Build uniform routing distribution: each expert processes the same number of tokens, and each token's top-k experts are unique."""
    assert top_k <= num_experts, "top_k cannot be greater than num_experts"

    # Generate a uniform distribution of expert IDs where each token has unique experts
    topk_ids = torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        topk_ids[i] = torch.arange(i * top_k, (i + 1) * top_k, dtype=torch.int32, device=device) % num_experts

    # Uniform weights
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32, device=device) / top_k

    return topk_weights, topk_ids


def benchmark_final():
    """Final MoE performance report."""
    device = "cuda"
    dtype = torch.bfloat16

    # Test configurations
    configs = [
        # (num_tokens, num_experts, top_k, hidden_size, ffn_size, label)
        (512, 64, 8, 1024, 128, "decode-small"),
        (4096, 64, 8, 1024, 128, "prefill-small"),
        (512, 128, 8, 2048, 128, "decode-medium"),
        (4096, 128, 8, 2048, 128, "prefill-medium"),
    ]

    activations = ["silu_and_mul", "gelu_and_mul"]

    print("=" * 120)
    print("TileOPs MoE Routed Expert Performance Report")
    print("=" * 120)
    print()
    print(f"Environment: GPU={torch.cuda.get_device_name()}, dtype={dtype}")
    print()

    all_results = []

    for activation in activations:
        print(f"\n{'='*120}")
        print(f"Activation: {activation}")
        print(f"{'='*120}\n")

        results = []

        for num_tokens, num_experts, top_k, hidden_size, ffn_size, label in configs:
            print(f"Configuration: {label}")
            print(f"  num_tokens={num_tokens}, num_experts={num_experts}, top_k={top_k}")
            print(f"  hidden_size={hidden_size}, ffn_size={ffn_size}")

            # Create inputs
            torch.manual_seed(42)
            hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            w_gate_up = torch.randn(num_experts, ffn_size * 2, hidden_size, dtype=dtype, device=device)
            w_down = torch.randn(num_experts, hidden_size, ffn_size, dtype=dtype, device=device)
            topk_weights, topk_ids = _build_uniform_routing(num_tokens, num_experts, top_k, device)

            # Calculate FLOPs (complete MoE pipeline)
            flops_gate_up = num_tokens * top_k * (2 * ffn_size) * hidden_size * 2
            flops_down = num_tokens * top_k * hidden_size * ffn_size * 2
            total_flops = flops_gate_up + flops_down

            result = {
                'config': label,
                'activation': activation,
                'num_tokens': num_tokens,
                'num_experts': num_experts,
                'top_k': top_k,
                'hidden_size': hidden_size,
                'ffn_size': ffn_size,
                'total_flops': total_flops,
            }

            # TileOPs outputs
            output_tileops = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
            workspace1 = torch.empty(0, dtype=dtype, device=device)
            workspace2 = torch.empty(0, dtype=dtype, device=device)

            # 1. TileOPs Separate
            op_separate = FusedMoEExpertsNopadPersistent3WGFwdOp(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                dtype=dtype,
                activation=activation,
                use_fused_activation=False,
            )

            for _ in range(5):
                op_separate(output_tileops, hidden_states, w_gate_up, w_down,
                           topk_weights, topk_ids, None, workspace1, workspace2, num_experts)

            def _run_separate():  # noqa: B023
                return op_separate(output_tileops, hidden_states, w_gate_up, w_down,  # noqa: B023
                                   topk_weights, topk_ids, None, workspace1, workspace2, num_experts)  # noqa: B023

            time_sep = _bench(_run_separate)
            tflops_sep = total_flops / time_sep * 1e-9

            result['separate_ms'] = time_sep
            result['separate_tflops'] = tflops_sep

            # 2. TileOPs Fused
            op_fused = FusedMoEExpertsNopadPersistent3WGFwdOp(
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                dtype=dtype,
                activation=activation,
                use_fused_activation=True,
            )

            for _ in range(5):
                op_fused(output_tileops, hidden_states, w_gate_up, w_down,
                        topk_weights, topk_ids, None, workspace1, workspace2, num_experts)

            def _run_fused():  # noqa: B023
                return op_fused(output_tileops, hidden_states, w_gate_up, w_down,  # noqa: B023
                                topk_weights, topk_ids, None, workspace1, workspace2, num_experts)  # noqa: B023

            time_fused = _bench(_run_fused)
            tflops_fused = total_flops / time_fused * 1e-9

            result['fused_ms'] = time_fused
            result['fused_tflops'] = tflops_fused
            result['speedup'] = time_sep / time_fused

            # 3. CUTLASS reference
            tokens_per_expert = (num_tokens * top_k) // num_experts
            cutlass_key = (tokens_per_expert, 2 * ffn_size, hidden_size, num_experts)
            cutlass_tflops = _CUTLASS_GROUPED_GEMM_TFLOPS.get(cutlass_key)
            result['cutlass_tflops'] = cutlass_tflops

            print(f"  Separate: {time_sep:.3f} ms, {tflops_sep:.2f} TFLOPS")
            print(f"  Fused:    {time_fused:.3f} ms, {tflops_fused:.2f} TFLOPS")
            print(f"  Speedup:  {result['speedup']:.2f}x")
            if cutlass_tflops:
                print(f"  CUTLASS:  {cutlass_tflops} TFLOPS (gate_up GEMM only)")
            print()

            results.append(result)

        all_results.extend(results)

        # Summary per activation
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"Average speedup ({activation}): {avg_speedup:.2f}x\n")

    # Final summary table
    print("=" * 120)
    print("Complete Summary Table")
    print("=" * 120)
    print()
    print(f"{'Config':<20} {'Activation':<15} {'Separate':<15} {'Fused':<15} {'Speedup':<10} {'CUTLASS*':<15}")
    print("-" * 120)

    for r in all_results:
        cutlass_str = f"{r['cutlass_tflops']}" if r['cutlass_tflops'] else "N/A"
        print(f"{r['config']:<20} {r['activation']:<15} "
              f"{r['separate_ms']:.3f} ms      {r['fused_ms']:.3f} ms      "
              f"{r['speedup']:.2f}x      {cutlass_str:<15}")

    print()
    print("* CUTLASS data only includes grouped GEMM (gate_up), not the full MoE pipeline")
    print()

    # Average speedup grouped by scenario
    print("=" * 120)
    print("Average Speedup by Scenario")
    print("=" * 120)
    print()

    decode_results = [r for r in all_results if 'decode' in r['config']]
    prefill_results = [r for r in all_results if 'prefill' in r['config']]
    silu_results = [r for r in all_results if r['activation'] == 'silu_and_mul']
    gelu_results = [r for r in all_results if r['activation'] == 'gelu_and_mul']

    print(f"Decode average speedup:  {sum(r['speedup'] for r in decode_results) / len(decode_results):.2f}x")
    print(f"Prefill average speedup: {sum(r['speedup'] for r in prefill_results) / len(prefill_results):.2f}x")
    print(f"SiLU average speedup: {sum(r['speedup'] for r in silu_results) / len(silu_results):.2f}x")
    print(f"GELU average speedup: {sum(r['speedup'] for r in gelu_results) / len(gelu_results):.2f}x")
    print(f"Overall average speedup:         {sum(r['speedup'] for r in all_results) / len(all_results):.2f}x")
    print()

    print("=" * 120)
    print("Benchmark completed!")
    print("=" * 120)


if __name__ == "__main__":
    benchmark_final()
