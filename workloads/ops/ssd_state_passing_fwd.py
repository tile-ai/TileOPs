import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdStatePassingFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
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
