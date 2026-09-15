"""Price the existing W4 packing without dequantization or GEMV.

Run from the repository root with ``python -m scripts.benchmark_w4a16_weight_stream``.
The checksums keep every weight read observable. Both probes use the benchmark
suite's CUPTI timing and cold-L2 protocol; neither is an Op performance baseline.
"""

import argparse
import json

import tilelang
import tilelang.language as T
import torch

from benchmarks.timing import bench_kernel, median_busy_ms


@tilelang.jit(out_idx=[-1])
def _linear_stream(size: int, block: int = 4096):
    @T.prim_func
    def main(
        weight: T.Tensor((size,), "int32"), output: T.Tensor((T.ceildiv(size, block),), "int32")
    ):
        with T.Kernel(T.ceildiv(size, block), threads=128) as bx:
            values = T.alloc_fragment((block,), "int32")
            total = T.alloc_fragment((1,), "int32")
            T.copy(weight[bx * block : (bx + 1) * block], values)
            T.reduce_sum(values, total, dim=0)
            output[bx] = total[0]

    return main


@tilelang.jit(out_idx=[-1])
def _tiled_stream(n: int, k: int):
    # Match the previous decode kernel's N/K tile and pipeline depth. This
    # separates the cost of its staged load pattern from unpacking and GEMV.
    block_n, block_k = 32, 512

    @T.prim_func
    def main(weight: T.Tensor((n, k // 2), "uint8"), output: T.Tensor((n,), "int32")):
        with T.Kernel(T.ceildiv(n, block_n), threads=128) as bx:
            packed = T.alloc_shared((block_n, block_k // 2), "uint8")
            values = T.alloc_fragment((block_n, block_k // 8), "int32")
            total = T.alloc_fragment((block_n,), "int32")
            T.clear(values)
            for kk in T.Pipelined(k // block_k, num_stages=4):
                T.copy(
                    weight[
                        bx * block_n : (bx + 1) * block_n,
                        kk * block_k // 2 : (kk + 1) * block_k // 2,
                    ],
                    packed,
                )
                for i, j in T.Parallel(block_n, block_k // 8):
                    for v in T.unroll(4):
                        values[i, j] += T.cast(packed[i, j * 4 + v], "int32")
            T.reduce_sum(values, total, dim=1)
            T.copy(total, output[bx * block_n : (bx + 1) * block_n])

    return main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=81920)
    args = parser.parse_args()
    if args.n <= 0 or args.n % 32 or args.k <= 0 or args.k % 512:
        parser.error("N must be a positive multiple of 32 and K a positive multiple of 512")
    torch.manual_seed(2130)
    packed = torch.randint(0, 256, (args.n, args.k // 2), device="cuda", dtype=torch.uint8)
    # A view of the supplied bytes, with no repacking or materialized conversion.
    words = packed.view(torch.int32).flatten()
    linear = _linear_stream(words.numel())
    tiled = _tiled_stream(args.n, args.k)
    torch.testing.assert_close(linear(words).sum().int(), words.sum().int())
    torch.testing.assert_close(tiled(packed).long(), packed.sum(dim=1))
    for name, kernel, tensor in (("linear", linear, words), ("tiled", tiled, packed)):
        samples = bench_kernel(kernel, args=(tensor,))
        milliseconds = median_busy_ms(samples)
        print(
            json.dumps(
                {
                    "probe": name,
                    "shape": [args.n, args.k],
                    "device": torch.cuda.get_device_name(),
                    "device_busy_ms": milliseconds,
                    "weight_read_bytes": packed.numel(),
                    "weight_read_bandwidth_tbs": packed.numel() / milliseconds / 1e9,
                    "samples": len(samples),
                }
            )
        )


if __name__ == "__main__":
    main()
