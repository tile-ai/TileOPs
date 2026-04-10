import torch

from tests.test_base import TestBase
from tileops.ops.ssd_chunk_scan import SsdChunkScanFwdOp
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdFixture
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdTest as _SsdChunkScanFwdTestWorkload


def ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt):
    """Triton-aligned PyTorch reference for chunk scan."""
    b, c, L, h, p = x.shape

    y_off = torch.einsum("bclhn,bchnp->bclhp", C.float(), prev_states.float())
    a_l = dA_cumsum.permute(0, 2, 3, 1).unsqueeze(-1)
    y_off = y_off * torch.exp(a_l)

    a_lhs = dA_cumsum.unsqueeze(-1)
    a_rhs = dA_cumsum.unsqueeze(-2)
    decay = torch.exp(a_lhs - a_rhs)
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask, 0)
    decay = decay.permute(0, 2, 1, 3, 4)
    dt_s = dt.float().permute(0, 1, 3, 2).unsqueeze(-2)
    lcb = cb.float() * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x.float())

    return y_off + y_diag


class SsdChunkScanFwdTest(_SsdChunkScanFwdTestWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_torch(x, cb, dA_cumsum, C, prev_states, dt)


@SsdChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune):
    test = SsdChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)
