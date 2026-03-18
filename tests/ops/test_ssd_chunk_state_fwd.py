import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.ssd_chunk_state_fwd import SsdChunkStateFwdOp


def ssd_chunk_state_fwd_ref(
    x: torch.Tensor,          # (b, seq_len, h, p)   flat sequence
    Bmat: torch.Tensor,       # (b, seq_len, g, n)
    dt: torch.Tensor,         # (b, h, c, Q)         float32
    dA_cumsum: torch.Tensor,  # (b, h, c, Q)         float32
    n_groups: int,
) -> torch.Tensor:
    """
    PyTorch reference for ssd_chunk_state_fwd.

    Returns:
      out: (b, c, h, n, p)  float32

    Semantics:
      out[b, c, h, n, p] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]

    where g(h) = h // heads_per_group.
    """
    b, seq_len, h, p = x.shape
    _, _, c, Q = dt.shape
    n = Bmat.shape[-1]
    heads_per_group = h // n_groups

    # Reshape flat sequence into chunked view
    # x:    (b, seq_len, h, p) -> (b, c, Q, h, p)
    # Bmat: (b, seq_len, g, n) -> (b, c, Q, g, n)
    x_chunked = x.float().reshape(b, c, Q, h, p)
    B_chunked = Bmat.float().reshape(b, c, Q, n_groups, n)

    # Map B from groups to heads: (b, c, Q, h, n)
    # Each head h belongs to group h // heads_per_group
    B_heads = B_chunked[:, :, :, torch.arange(h) // heads_per_group, :]

    # dA_cumsum: (b, h, c, Q) -> (b, c, h, Q)
    dA = dA_cumsum.float().permute(0, 2, 1, 3)

    # dA_end: (b, c, h, 1)  -- last position in each chunk
    dA_end = dA[:, :, :, -1:]

    # decay[b, c, h, l] = exp(dA_end - dA[l])  shape: (b, c, h, Q)
    decay = torch.exp(torch.clamp(dA_end - dA, max=0.0))

    # dt: (b, h, c, Q) -> (b, c, h, Q)
    dt_chunked = dt.float().permute(0, 2, 1, 3)

    # weight[b, c, h, l] = decay[l] * dt[l]
    weight = decay * dt_chunked  # (b, c, h, Q)

    # out[b, c, h, n, p] = sum_l weight[b,c,h,l] * B[b,c,l,h,n] * x[b,c,l,h,p]
    # weight: (b, c, h, Q) -> (b, c, Q, h, 1, 1) for broadcasting
    w = weight.permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)  # (b, c, Q, h, 1, 1)
    # B_heads: (b, c, Q, h, n) -> (b, c, Q, h, n, 1)
    # x_chunked: (b, c, Q, h, p) -> (b, c, Q, h, 1, p)
    contrib = w * B_heads.unsqueeze(-1) * x_chunked.unsqueeze(-2)  # (b, c, Q, h, n, p)

    # sum over l (Q dimension)
    out = contrib.sum(dim=2)  # (b, c, h, n, p)
    return out


class SsdChunkStateFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune", [
            (1, 2,  64, 4,  64, 32, 1, torch.float16,  False),
            (2, 4,  64, 8,  64, 64, 2, torch.float16,  False),
            (1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False),
            (2, 2,  64, 4,  64, 32, 2, torch.bfloat16, False),
        ]),
    ]


class SsdChunkStateFwdTest(TestBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, c, Q, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        seq_len = c * Q
        x = torch.randn(b, seq_len, h, p, dtype=self.dtype, device="cuda") * 0.1
        Bmat = torch.randn(b, seq_len, g, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: monotonically non-decreasing (negative values, cumsum of negatives)
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        return x, Bmat, dt, dA_cumsum

    def ref_program(self, x, Bmat, dt, dA_cumsum):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups)


@SsdChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune,
):
    test = SsdChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
    )
    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-1 if dtype == torch.float16 else 2e-1
    rtol = 1e-1
    test.check(op, *inputs, atol=atol, rtol=rtol)
