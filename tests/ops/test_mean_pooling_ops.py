import argparse
from top.ops import MeanPoolingForwardOp
from top.utils import str2dtype
from benchmarks.deepseek_nsa.deepseek_nsa import MeanPoolingForwardBenchmark


def test_mean_pooling_op( 
        batch_size,
        total_seqlen,
        total_chunks,
        heads,
        dim,
        chunk_size,
        tune= True
):
    op = MeanPoolingForwardOp(batch_size, total_seqlen, total_chunks, heads, dim, chunk_size, tune=tune)
    
    benchmark = MeanPoolingForwardBenchmark(batch_size, total_seqlen, total_chunks, heads, dim, chunk_size,tune=tune)

    inputs = benchmark.gen_inputs()
    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)
    benchmark.baseline_profile(*inputs)

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='logical batch size')
    parser.add_argument('--total_seqlen', type=int, default=1*8192*1, help='number of heads')
    parser.add_argument('--total_chunks', type=int, default=1*256*1, help='sequence length')
    parser.add_argument('--heads', type=int, default=128, help='head dim')
    parser.add_argument('--dim', type=int, default=128, help='scale')
    parser.add_argument('--chunk_size', type=int, default=32, help='scale')
    parser.add_argument('--tune', action='store_true', default=True, help='enable autotune')
    args = parser.parse_args()
    test_mean_pooling_op(
        args.batch_size,
        args.total_seqlen,
        args.total_chunks,
        args.heads,
        args.dim,
        args.chunk_size,
        args.tune,
    )