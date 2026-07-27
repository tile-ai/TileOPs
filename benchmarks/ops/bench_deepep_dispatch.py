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

from tileops.ops.moe import DeepEPDispatchAdapter

WARMUP = 10
ITERS = 50


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
        hidden = torch.randn(
            num_tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk_ids = (
            torch.rand(num_tokens, args.num_experts, device="cuda")
            .topk(args.top_k, dim=-1)
            .indices.to(torch.int64)
        )
        topk_weights = torch.softmax(torch.randn(num_tokens, args.top_k, device="cuda"), dim=-1)
        offsets = torch.empty(
            args.num_experts // world_size + 1,
            dtype=torch.int32,
            device="cuda",
        )

        def run(
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

        result = run()
        dispatch_fresh_ms = _time_ms(run)
        cached_offsets = torch.empty_like(offsets)

        def run_cached(
            hidden=hidden,
            topk_weights=topk_weights,
            offsets=cached_offsets,
            cached_handle=result.combine_handle,
        ):
            return adapter.dispatch(
                hidden,
                None,
                topk_weights,
                expert_offsets=offsets,
                cached_handle=cached_handle,
            )

        cached_result = run_cached()
        dispatch_cached_ms = _time_ms(run_cached)
        torch.testing.assert_close(
            cached_result.batch.expert_offsets,
            result.batch.expert_offsets,
        )
        valid_rows = int(result.batch.valid_rows.item())
        physical_rows = result.batch.capacity
        sent_pairs = num_tokens * args.top_k
        record = {
            "rank": rank,
            "world_size": world_size,
            "tokens": num_tokens,
            "top_k": args.top_k,
            "num_experts": args.num_experts,
            "num_local_experts": args.num_experts // world_size,
            "hidden_size": args.hidden_size,
            "sent_pairs": sent_pairs,
            "received_pairs": valid_rows,
            "physical_rows": physical_rows,
            "dispatch_fresh_allocating_ms": round(dispatch_fresh_ms, 4),
            "dispatch_cached_allocating_ms": round(dispatch_cached_ms, 4),
        }
        print(json.dumps(record), flush=True)
        dist.barrier(group)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
