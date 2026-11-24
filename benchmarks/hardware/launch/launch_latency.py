"""Kernel Launch Latency Benchmark — measures launch.latency in src/tileops/_perf/profiles/.

The device-busy time one more dependent kernel launch adds to a program. Each program
runs ``k`` one-block kernels in sequence, each reading the value the previous one
wrote, for ``k`` = 1 .. ``--max-launches``. Every program is timed as TileOPs times
tuning candidates (CUPTI device busy, L2 flushed before each call, median of the
repeats). One step is the busy time from ``k`` to ``k + 1`` launches; the latency is the
median step over all runs, a typical cost rather than an envelope
(docs/design/roofline.md §5.1).

Usage:
    python benchmarks/hardware/launch/launch_latency.py [--profile h200] [--runs 5]
"""

import argparse
import statistics

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.autotune import candidate_busy_ms

_LANES = 128


def _program(launches):
    """A program of *launches* one-block kernels, each reading what the previous one wrote."""

    @tilelang.jit(out_idx=[])
    def build():
        @T.prim_func
        def main(a: T.Tensor((_LANES,), "float32"), b: T.Tensor((_LANES,), "float32")):
            for step in T.unroll(launches):
                with T.Kernel(1, threads=_LANES) as _bx:
                    for t in T.Parallel(_LANES):
                        if step % 2 == 0:
                            b[t] = a[t] + 1.0
                        else:
                            a[t] = b[t] + 1.0

        return main

    return build()


def main():
    parser = argparse.ArgumentParser(description="Kernel launch latency microbenchmark")
    parser.add_argument("--profile", default="h200", help="GPU profile name")
    parser.add_argument("--runs", type=int, default=5, help="Independent runs over all chains")
    parser.add_argument("--max-launches", type=int, default=5, help="Longest chain timed")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup calls per program")
    parser.add_argument("--repeat", type=int, default=200, help="Timed calls per program")
    args = parser.parse_args()

    print(f"Profile: {args.profile}")
    print(f"Device:  {torch.cuda.get_device_name()}")
    print(
        f"Chains of 1..{args.max_launches} launches, {args.runs} runs, "
        f"median of {args.repeat} calls each\n"
    )

    a = torch.zeros(_LANES, device="cuda")
    b = torch.zeros(_LANES, device="cuda")
    lengths = range(1, args.max_launches + 1)
    programs = {k: _program(k) for k in lengths}

    steps = []
    for run in range(args.runs):
        busy_us = {
            k: candidate_busy_ms(lambda k=k: programs[k](a, b), args.warmup, args.repeat) * 1e3
            for k in lengths
        }
        step = [busy_us[k + 1] - busy_us[k] for k in lengths[:-1]]
        steps.extend(step)
        print(
            f"run {run}: busy us "
            + " ".join(f"{busy_us[k]:.3f}" for k in lengths)
            + " | step us "
            + " ".join(f"{s:.3f}" for s in step),
            flush=True,
        )

    latency_us = statistics.median(steps)
    print(f"\n{'=' * 60}")
    print(f"Steps:            {len(steps)}, {min(steps):.3f} - {max(steps):.3f} us")
    print(f"Median step:      {latency_us:.3f} us")
    print(f"\nUpdate src/tileops/_perf/profiles/{args.profile}.yaml:")
    print(f"  launch.latency: {latency_us:.3f}e-6")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
