import torch

from tests.test_base import TestBase
from tileops.ops.ssd_chunk_state_fwd import SsdChunkStateFwdOp
from workloads.ops.ssd_chunk_state_fwd import SsdChunkStateFwdFixture, ssd_chunk_state_fwd_ref
from workloads.ops.ssd_chunk_state_fwd import SsdChunkStateFwdTest as _SsdChunkStateFwdTestWorkload


class SsdChunkStateFwdTest(_SsdChunkStateFwdTestWorkload, TestBase):
    def ref_program(self, x, Bmat, dt, dA_cumsum, seq_idx):
        return ssd_chunk_state_fwd_ref(x, Bmat, dt, dA_cumsum, self.n_groups, seq_idx=seq_idx)


@SsdChunkStateFwdFixture
def test_ssd_chunk_state_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx,
):
    test = SsdChunkStateFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, has_seq_idx,
    )
    op = SsdChunkStateFwdOp(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        has_seq_idx=has_seq_idx, tune=tune,
    )
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    test.check(op, *inputs, atol=atol, rtol=rtol)
