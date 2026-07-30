"""Single-node multi-GPU DeepEP V2 dispatch benchmark for the M6 adapter.

Example:
    torchrun --standalone --nproc-per-node=8 \
        benchmarks/ops/bench_deepep_dispatch.py \
        --tokens 1 8 32 128 --hidden-size 7168 --num-experts 256 --top-k 8

Only dispatch is timed.  Expert compute and combine are deliberately excluded.
The script requires DeepEP V2, but TileOps itself does not depend on DeepEP.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from benchmarks.ops._moe_bench_utils import (
    make_routing_inputs,
    measure_dispatch,
)
from tileops.ops.moe import DeepEPDispatchAdapter

WARMUP = 10
ITERS = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--num-sms", type=int, default=0)
    args = parser.parse_args()

    try:
        from deep_ep import ElasticBuffer
    except ImportError as exc:
        raise RuntimeError("bench_deepep_dispatch.py requires a DeepEP V2 installation") from exc

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if args.num_experts % world_size != 0:
        raise ValueError(
            f"num_experts must be divisible by world_size; got {args.num_experts} and {world_size}"
        )
    if args.top_k > args.num_experts:
        raise ValueError("top_k cannot exceed num_experts")

    max_tokens = max(args.tokens)
    buffer = ElasticBuffer(
        group,
        num_max_tokens_per_rank=max_tokens,
        hidden=args.hidden_size,
        num_topk=args.top_k,
        use_fp8_dispatch=False,
    )
    adapter = DeepEPDispatchAdapter(
        buffer,
        num_experts=args.num_experts,
        num_local_experts=args.num_experts // world_size,
        num_max_tokens_per_rank=max_tokens,
        num_sms=args.num_sms,
    )

    for num_tokens in args.tokens:
        torch.manual_seed(2026 + rank)
        hidden, topk_ids, topk_weights = make_routing_inputs(
            num_tokens,
            args.hidden_size,
            args.top_k,
            args.num_experts,
            topk_dtype=torch.int64,
        )
        offsets = torch.empty(
            args.num_experts // world_size + 1,
            dtype=torch.int32,
            device="cuda",
        )

        def run_fresh(
            hidden=hidden,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            offsets=offsets,
        ):
            return adapter.dispatch(
                hidden,
                topk_ids,
                topk_weights,
                expert_offsets=offsets,
            )

        def make_cached(
            result,
            offsets=offsets,
            cached_hidden=hidden,
            cached_weights=topk_weights,
        ):
            cached_offsets = torch.empty_like(offsets)

            def run_cached():
                return adapter.dispatch(
                    cached_hidden,
                    None,
                    cached_weights,
                    expert_offsets=cached_offsets,
                    cached_handle=result.combine_handle,
                )

            return run_cached

        def validate_cached(result, cached_result):
            torch.testing.assert_close(
                cached_result.batch.expert_offsets,
                result.batch.expert_offsets,
            )

        result, fresh_ms, cached_ms = measure_dispatch(
            run_fresh,
            warmup=WARMUP,
            iters=ITERS,
            make_cached=make_cached,
            validate_cached=validate_cached,
        )
        assert cached_ms is not None
        record = {
            "rank": rank,
            "world_size": world_size,
            "tokens": num_tokens,
            "top_k": args.top_k,
            "num_experts": args.num_experts,
            "num_local_experts": args.num_experts // world_size,
            "hidden_size": args.hidden_size,
            "sent_pairs": num_tokens * args.top_k,
            "received_pairs": int(result.batch.valid_rows.item()),
            "physical_rows": result.batch.capacity,
            "dispatch_fresh_allocating_ms": round(fresh_ms, 4),
            "dispatch_cached_allocating_ms": round(cached_ms, 4),
        }
        print(json.dumps(record), flush=True)
        dist.barrier(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
