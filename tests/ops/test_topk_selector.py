import pytest
from benchmarks import TopkSelectorBenchmark
from top.ops import TopkSelectorOp
from top.utils import str2dtype


@pytest.mark.parametrize("batch, seq_len, seq_len_kv, topk, in_dtype, out_dtype, tune", [
    (64, 1024, 32 * 1024, 1024, "float32", "int32", False),
    (64, 1024, 32 * 1024, 2048, "float32", "int32", False),
    (128, 1024, 64 * 1024, 1024, "float32", "int32", False),
    (128, 1024, 64 * 1024, 2048, "float32", "int32", False),
])
def test_topk_selector_op(batch: int, seq_len: int, seq_len_kv: int, topk: int, in_dtype: str,
                          out_dtype: str, tune: bool) -> None:
    in_dtype = str2dtype[in_dtype]
    out_dtype = str2dtype[out_dtype]

    op = TopkSelectorOp(batch, seq_len, seq_len_kv, topk, in_dtype, out_dtype, tune=tune)
    benchmark = TopkSelectorBenchmark(batch, seq_len, seq_len_kv, topk, in_dtype, out_dtype)

    inputs = benchmark.gen_inputs()

    benchmark.check(op, *inputs, atol=1e-4, rtol=1e-5)
    benchmark.profile(op, *inputs)


if __name__ == "__main__":
    test_topk_selector_op(64, 1024, 32 * 1024, 1024, "float32", "int32", False)
    test_topk_selector_op(64, 1024, 32 * 1024, 2048, "float32", "int32", False)
    test_topk_selector_op(128, 1024, 64 * 1024, 1024, "float32", "int32", False)
    test_topk_selector_op(128, 1024, 64 * 1024, 2048, "float32", "int32", False)
