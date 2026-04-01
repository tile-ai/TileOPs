import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.da_cumsum_fwd import DaCumsumFwdOp


def da_cumsum_fwd_ref(
    dt: torch.Tensor,   # (b, seq_len, h)  float32
    A: torch.Tensor,    # (h,)             float32
    num_chunks: int,
    chunk_len: int,
) -> torch.Tensor:
    """
    PyTorch reference for da_cumsum_fwd.

    Returns:
      dA_cumsum: (b, h, c, Q)  float32

    Semantics:
      dA_cumsum[b, h, c, l] = sum_{i=0}^{l} dt[b, c*Q+i, h] * A[h]

    This matches A_cumsum in the Mamba-2 ssd_minimal_discrete reference:
      A = rearrange(dt * A_log, "b (c l) h -> b h c l")
      A_cumsum = torch.cumsum(A, dim=-1)
    """
    b, S, h = dt.shape
    Q = chunk_len
    C = num_chunks

    # (b, S, h) -> (b, C, Q, h)
    dt_chunked = dt.float().reshape(b, C, Q, h)
    # dA = dt * A, broadcast A: (h,) -> (1, 1, 1, h)
    dA = dt_chunked * A.float()          # (b, C, Q, h)
    # Inclusive prefix sum along the position axis (dim=2)
    dA_cumsum = dA.cumsum(dim=2)         # (b, C, Q, h)
    # Rearrange to match output layout (b, h, C, Q)
    return dA_cumsum.permute(0, 3, 1, 2).contiguous()


class DaCumsumFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, tune", [
            pytest.param(1, 2, 64, 4, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 64, 8, False, marks=pytest.mark.full),
            pytest.param(1, 2, 128, 4, False, marks=pytest.mark.full),
            pytest.param(2, 4, 128, 16, False, marks=pytest.mark.full),
        ]),
    ]


class DaCumsumFwdTest(TestBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads

    def gen_inputs(self):
        b, C, Q, h = self.batch, self.num_chunks, self.chunk_len, self.n_heads
        seq_len = C * Q
        # dt > 0 (softplus output in Mamba-2), A <= 0 (negative decay)
        dt = torch.rand(b, seq_len, h, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        A = -torch.rand(h, dtype=torch.float32, device="cuda")
        return dt, A

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
