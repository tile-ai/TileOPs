from typing import Optional

from benchmarks.benchmark import BenchmarkBase


class TopkSelectorBenchmark(BenchmarkBase):

    def calculate_flops(self) -> Optional[float]:
        return None

    def calculate_memory(self) -> Optional[float]:
        t = self.test
        index_score_memory = (t.batch * t.seq_len * t.seq_len_kv * t.kv_group * t.in_dtype.itemsize)
        index_memory = t.batch * t.seq_len * t.topk * t.kv_group * t.out_dtype.itemsize
        starts_memory = t.batch * t.seq_len * t.out_dtype.itemsize
        ends_memory = t.batch * t.seq_len * t.out_dtype.itemsize
        return index_score_memory + index_memory + starts_memory + ends_memory
