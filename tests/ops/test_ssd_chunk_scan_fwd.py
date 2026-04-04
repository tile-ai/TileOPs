import torch

from tests.test_base import TestBase
from tileops.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdOp
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdFixture, ssd_chunk_scan_fwd_ref
from workloads.ops.ssd_chunk_scan_fwd import SsdChunkScanFwdTest as _SsdChunkScanFwdTestWorkload


class SsdChunkScanFwdTest(_SsdChunkScanFwdTestWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt)


@SsdChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune):
    test = SsdChunkScanFwdTest(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype)
    op = SsdChunkScanFwdOp(batch, num_chunks, chunk_len, n_heads, d_head, d_state, dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)
