import torch

from tests.test_base import TestBase
from tileops.ops.da_cumsum import DaCumsumFwdOp
from workloads.ops.da_cumsum_fwd import DaCumsumFwdFixture
from workloads.ops.da_cumsum_fwd import DaCumsumFwdTest as _DaCumsumFwdTestWorkload


def da_cumsum_fwd_ref(
    dt: torch.Tensor,
    A: torch.Tensor,
    num_chunks: int,
    chunk_len: int,
) -> torch.Tensor:
    """PyTorch reference for da_cumsum_fwd."""
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks
    dt_chunked = dt.float().reshape(b, C, Q, h)
    dA = dt_chunked * A.float()
    dA_cumsum = dA.cumsum(dim=2)
    return dA_cumsum.permute(0, 3, 1, 2).contiguous()


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
