import inspect
import pytest
import torch

from benchmarks.deepseek_nsa.deepseek_nsa import NSACmpFwdVarlenBenchmark
from top.ops import NSACmpFwdVarlenOp


@pytest.mark.parametrize(
    ("seq_num, c_seq_len, heads, dim_k, dim_v, group, scale, bc, bs, bk, bv, "
     "dtype, accum_dtype, tune"),
    [
        (9, 8192, 32, 128, 128, 16, 128**
         -0.5, 32, 32, 128, 128, torch.float16, torch.float32, False),
    ],
)
def test_nsa_cmp_fwd_varlen_op(
    seq_num: int,
    c_seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    group: int,
    scale: float,
    bc: int,
    bs: int,
    bk: int,
    bv: int,
    dtype: torch.dtype,
    accum_dtype: torch.dtype,
    tune: bool,
) -> None:

    assert group % 16 == 0, "Group size must be a multiple of 16 in NSA"

    # Create params dictionary from function arguments using the function signature
    # to avoid including pytest-injected local variables.
    # Note: Need to capture locals() before list comprehension due to scope issues
    local_vars = locals()
    sig = inspect.signature(globals()[inspect.stack()[0].function])
    params = {name: local_vars[name] for name in sig.parameters}
    benchmark = NSACmpFwdVarlenBenchmark(**params)
    inputs = benchmark.gen_inputs()

    params["chunk_num"] = benchmark.chunk_num
    op = NSACmpFwdVarlenOp(**params)
    benchmark.check(op, *inputs, atol=4e-3, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
