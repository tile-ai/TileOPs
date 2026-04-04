import pytest
import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdStatePassingFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, num_chunks, n_heads, d_state, dtype, tune", [
            pytest.param(1, 2,  4,  32, torch.float16,  False, marks=pytest.mark.smoke),
            pytest.param(2, 4,  8,  64, torch.float16,  False, marks=pytest.mark.full),
            pytest.param(1, 2,  4,  32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 4,  8,  64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


class SsdStatePassingFwdTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.dtype = dtype

    def gen_inputs(self):
        b, c, h, d = self.batch, self.num_chunks, self.n_heads, self.d_state
        states = torch.randn(b, c, h, d, dtype=self.dtype, device="cuda") * 0.1
        dA_chunk_cumsum = -torch.rand(b, h, c, dtype=torch.float32, device="cuda").cumsum(-1)
        initial_states = torch.randn(b, h, d, dtype=torch.float32, device="cuda") * 0.1
        return states, dA_chunk_cumsum, initial_states

    def ref_program(self, states, dA_chunk_cumsum, initial_states):
        return ssd_state_passing_fwd_ref(states, dA_chunk_cumsum, initial_states)


def ssd_state_passing_fwd_ref(
    states: torch.Tensor,           # (b, c, h, d)
    dA_chunk_cumsum: torch.Tensor,  # (b, h, c)
    initial_states: torch.Tensor,   # (b, h, d)
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the inter-chunk recurrent scan.

    s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

    Returns:
        out:          (b, c, h, d) float32
        final_states: (b, h, d) float32
    """
    b, c, h, d = states.shape
    out = []
    s = initial_states.float()  # (b, h, d)

    for ci in range(c):
        scale = torch.exp(dA_chunk_cumsum[:, :, ci]).unsqueeze(-1)  # (b, h, 1)
        u = states[:, ci, :, :].float()                             # (b, h, d)
        s = scale * s + u
        out.append(s.clone())

    return torch.stack(out, dim=1), s  # (b, c, h, d), (b, h, d)
