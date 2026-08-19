"""Dispatch-only benchmark for the tight local M6 reference path.

DeepEP communication is intentionally not emulated here.  A real DeepEP run
must report communication and adapter normalization separately; this benchmark
measures the local count/prefix-sum/scatter/gather primitive used as the
world-size-one reference.
"""

import argparse

import torch

from benchmarks.ops._moe_bench_utils import (
    make_routing_inputs,
    measure_dispatch,
)
from tileops.ops.moe import LocalExpertDispatcher

WARMUP = 20
ITERS = 100
FIELDS = (
    "tokens",
    "top_k",
    "num_experts",
    "hidden_size",
    "distribution",
    "effective_rows",
    "physical_rows",
    "dispatch_allocating_ms",
    "effective_GBps",
)


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

    print(",".join(FIELDS))
    for num_tokens in args.tokens:
        for distribution in args.distributions:
            hidden, topk_ids, weights = make_routing_inputs(
                num_tokens,
                args.hidden_size,
                args.top_k,
                args.num_experts,
                distribution=distribution,
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

            result, fresh_ms, _ = measure_dispatch(run, warmup=WARMUP, iters=ITERS)
            effective_rows = num_tokens * args.top_k
            logical_bytes = 2 * effective_rows * args.hidden_size * hidden.element_size()
            values = (
                num_tokens,
                args.top_k,
                args.num_experts,
                args.hidden_size,
                distribution,
                effective_rows,
                result.batch.capacity,
                f"{fresh_ms:.4f}",
                f"{logical_bytes / (fresh_ms / 1e3) / 1e9:.2f}",
            )
            print(",".join(map(str, values)), flush=True)


if __name__ == "__main__":
    main()
