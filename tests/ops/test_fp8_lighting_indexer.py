import sys
from typing import Optional

import pytest

from benchmarks import Fp8LightingIndexerBenchmark
from top.ops import Fp8LightingIndexerOp


@pytest.mark.parametrize(
    "seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune",
    [
        (4096, 32, 64, 8192, True, None, False),
    ],
)
def test_indexer(seq_len: int, heads: int, index_dim: int, seq_len_kv: int, clean_logits: bool,
                 config: Optional[dict], tune: bool) -> None:
    op = Fp8LightingIndexerOp(
        seq_len, heads, index_dim, seq_len_kv, clean_logits, config, tune=tune)
    benchmark = Fp8LightingIndexerBenchmark(seq_len, heads, index_dim, seq_len_kv, clean_logits,
                                            config)

    inputs = benchmark.gen_inputs()

    benchmark.check(op, *inputs)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    errno = pytest.main([__file__, "-vvs"])
    sys.exit(errno)
