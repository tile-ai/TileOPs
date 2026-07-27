"""Dispatch-only benchmark for the tight local M6 reference path.

DeepEP communication is intentionally not emulated here.  A real DeepEP run
must report communication and adapter normalization separately; this benchmark
measures the local count/prefix-sum/scatter/gather primitive used as the
world-size-one reference.
"""

import argparse

import torch

from tileops.ops.moe import LocalExpertDispatcher

WARMUP = 20
ITERS = 100


def _time_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / ITERS


def _make_topk_ids(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    distribution: str,
) -> torch.Tensor:
    if distribution == "uniform":
        scores = torch.rand(num_tokens, num_experts, device="cuda")
    elif distribution == "hotspot":
        logits = torch.linspace(4.0, -4.0, num_experts, device="cuda")
        scores = logits.unsqueeze(0) + torch.rand(num_tokens, num_experts, device="cuda")
    elif distribution == "longtail":
        ranks = torch.arange(1, num_experts + 1, device="cuda")
        logits = -torch.log(ranks)
        scores = logits.unsqueeze(0) + torch.rand(num_tokens, num_experts, device="cuda")
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return scores.topk(top_k, dim=-1).indices.to(torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512, 2048])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "longtail", "hotspot"],
        default=["uniform", "longtail", "hotspot"],
    )
    args = parser.parse_args()

    print(
        "tokens,top_k,num_experts,hidden_size,distribution,"
        "effective_rows,physical_rows,dispatch_allocating_ms,effective_GBps"
    )
    for num_tokens in args.tokens:
        hidden = torch.randn(
            num_tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        weights = torch.softmax(torch.randn(num_tokens, args.top_k, device="cuda"), dim=-1)
        for distribution in args.distributions:
            topk_ids = _make_topk_ids(
                num_tokens,
                args.top_k,
                args.num_experts,
                distribution,
            )
            dispatcher = LocalExpertDispatcher(
                args.num_experts,
                total_tokens=num_tokens,
                top_k=args.top_k,
                hidden_size=args.hidden_size,
                dtype=torch.bfloat16,
            )

            def run(
                dispatcher=dispatcher,
                hidden=hidden,
                topk_ids=topk_ids,
                weights=weights,
            ):
                return dispatcher.dispatch(hidden, topk_ids, weights)

            result = run()
            torch.cuda.synchronize()
            dispatch_ms = _time_ms(run)
            effective_rows = num_tokens * args.top_k
            physical_rows = result.batch.capacity
            # One source read and one dispatched write for every routed row.
            logical_bytes = 2 * effective_rows * args.hidden_size * hidden.element_size()
            effective_gbps = logical_bytes / (dispatch_ms / 1e3) / 1e9
            print(
                f"{num_tokens},{args.top_k},{args.num_experts},"
                f"{args.hidden_size},{distribution},{effective_rows},"
                f"{physical_rows},{dispatch_ms:.4f},{effective_gbps:.2f}"
            )


if __name__ == "__main__":
    main()
