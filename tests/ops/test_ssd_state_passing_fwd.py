import torch

from tests.test_base import TestBase
from tileops.ops.ssd_state_passing import SsdStatePassingFwdOp
from workloads.ops.ssd_state_passing_fwd import SsdStatePassingFwdFixture
from workloads.ops.ssd_state_passing_fwd import (
    SsdStatePassingFwdTest as _SsdStatePassingFwdTestWorkload,
)


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,
    dA_chunk_cumsum: torch.Tensor,
    initial_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the inter-chunk recurrent scan."""
    b, c, h, d = states.shape
    out = []
    s = initial_states.float()

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)
        u = states[:, ci, :, :].float()
        s = scale * s + u
        out.append(s.clone())

    return torch.stack(out, dim=1), s


class SsdStatePassingFwdTest(_SsdStatePassingFwdTestWorkload, TestBase):
    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


@SsdStatePassingFwdFixture
def test_ssd_state_passing_fwd(batch, num_chunks, n_heads, d_state, dtype, tune):
    test = SsdStatePassingFwdTest(batch, num_chunks, n_heads, d_state, dtype)
    op = SsdStatePassingFwdOp(batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune)
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    test.check(op, *inputs, atol=atol, rtol=rtol)
