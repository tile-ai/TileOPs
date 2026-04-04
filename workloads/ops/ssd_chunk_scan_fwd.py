import pytest
import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdChunkScanFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune", [
            pytest.param(1, 2, 64, 4, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 64, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 2, 128, 4, 128, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 2, 64, 4, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]

class SsdChunkScanFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        chunk_len: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.dtype = dtype

    def gen_inputs(self):
        b, c, L, h, p, n = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state,
        )
        x = torch.randn(b, c, L, h, p, dtype=self.dtype, device="cuda") * 0.1
        cb = torch.randn(b, c, h, L, L, dtype=self.dtype, device="cuda") * 0.1
        dA_cumsum = torch.zeros(b, h, c, L, dtype=torch.float32, device="cuda")
        # fill with plausible negative cumsum values (decaying system)
        dA_cumsum = -torch.rand(b, h, c, L, dtype=torch.float32, device="cuda").cumsum(-1)
        C = torch.randn(b, c, L, h, n, dtype=self.dtype, device="cuda") * 0.1
        prev_states = torch.randn(b, c, h, n, p, dtype=self.dtype, device="cuda") * 0.1
        dt = torch.rand(b, c, L, h, dtype=self.dtype, device="cuda") * 0.1 + 0.01
        return x, cb, dA_cumsum, C, prev_states, dt

def ssd_chunk_scan_fwd_ref(
    x: torch.Tensor,           # (b, c, L, h, p)      raw x
    cb: torch.Tensor,          # (b, c, h, L, L)
    dA_cumsum: torch.Tensor,  # (b, h, c, L)         cumsum of dA, where dA = dt * A
    C: torch.Tensor,           # (b, c, L, h, n)
    prev_states: torch.Tensor, # (b, c, h, n, p)
    dt: torch.Tensor,          # (b, c, L, h)         per-token dt
) -> torch.Tensor:
    """
    Triton-aligned PyTorch reference for chunk scan.

    Returns:
      out: (b, c, L, h, p)

    Semantics:
      Step 4 / history path:
        y_off[l, p] = exp(dA_cumsum[l]) * sum_n C[l, n] * prev_states[n, p]

      Step 1 / intra-chunk diagonal path:
        y_diag[l, p] = sum_s cb[l, s]
                            * exp(dA_cumsum[l] - dA_cumsum[s])
                            * dt[s]
                            * x[s, p]

    Notes:
      - `x` is the raw input, NOT pre-multiplied by dt.
      - `dA_cumsum` must come from cumulative sum of dA = A * dt.
      - If you instead pass x_dt = x * dt.unsqueeze(-1), then you should NOT
        multiply dt again inside this function.
    """
    b, c, L, h, p = x.shape

    # -----------------------------------
    # Step 4 / history path
    # -----------------------------------
    # C:           (b, c, L, h, n)
    # prev_states: (b, c, h, n, p)
    # -> y_off:    (b, c, L, h, p)
    y_off = torch.einsum(
        "bclhn,bchnp->bclhp",
        C.float(),
        prev_states.float(),
    )

    # dA_cumsum: (b, h, c, L) -> (b, c, L, h, 1)
    a_l = dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1)
    y_off = y_off * torch.exp(a_l)

    # -----------------------------------
    # Step 1 / intra-chunk diagonal path
    # -----------------------------------
    # decay[l,s] = exp(dA_cs[l] - dA_cs[s]) if s <= l else 0
    a_lhs = dA_cumsum.unsqueeze(-1)   # (b, h, c, L, 1)
    a_rhs = dA_cumsum.unsqueeze(-2)   # (b, h, c, 1, L)
    decay = torch.exp(a_lhs - a_rhs)   # (b, h, c, L, L)

    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0)

    # align with cb: (b, c, h, L, L)
    decay = decay.permute(0, 2, 1, 3, 4)

    # dt is indexed by source timestep s, not target timestep l
    # dt: (b, c, L, h) -> (b, c, h, 1, L)
    dt_s = dt.float().permute(0, 1, 3, 2).unsqueeze(-2)

    # lcb[l,s] = cb[l,s] * decay[l,s] * dt[s]
    lcb = cb.float() * decay * dt_s

    # y_diag[l,p] = sum_s lcb[l,s] * x[s,p]
    y_diag = torch.einsum(
        "bchls,bcshp->bclhp",
        lcb,
        x.float(),
    )

    return y_off + y_diag
