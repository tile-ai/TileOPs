
from tests.test_base import TestBase
from tileops.ops.da_cumsum_fwd import DaCumsumFwdOp
from workloads.ops.da_cumsum_fwd import DaCumsumFwdFixture, da_cumsum_fwd_ref
from workloads.ops.da_cumsum_fwd import DaCumsumFwdTest as _DaCumsumFwdTestWorkload


class DaCumsumFwdTest(_DaCumsumFwdTestWorkload, TestBase):
    def ref_program(self, dt, A):
        return da_cumsum_fwd_ref(dt, A, self.num_chunks, self.chunk_len)


@DaCumsumFwdFixture
def test_da_cumsum_fwd(batch, num_chunks, chunk_len, n_heads, tune):
    test = DaCumsumFwdTest(batch, num_chunks, chunk_len, n_heads)
    op = DaCumsumFwdOp(
        batch, num_chunks, chunk_len, n_heads,
        seq_len=num_chunks * chunk_len,
        tune=tune,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs, atol=1e-5, rtol=1e-5)
