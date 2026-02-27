from typing import Optional

from tests.ops.test_mha_decode_paged import MhaDecodePagedTest
from benchmarks.benchmark import BenchmarkBase, BenchmarkReport


class MhaDecodePagedBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        t = self.test
        flops_per_matmul = 2.0 * t.batch * t.heads * t.seqlen_q * t.seqlen_kv * t.dim
        flops = flops_per_matmul * 2
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        num_pages = t.seqlen_kv // t.page_size
        # Q, output: batch * seqlen_q * heads * dim; K,V: seqlen_kv * heads * dim; block_table, real_seqlen_kv: int32
        return (t.batch * t.seqlen_q * t.heads * t.dim * 2 +
                2 * t.seqlen_kv * t.heads * t.dim) * t.dtype.itemsize + \
            t.batch * num_pages * 4 + t.batch * 4
