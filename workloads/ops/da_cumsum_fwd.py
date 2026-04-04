import torch

from workloads.base import FixtureBase, WorkloadBase


class DaCumsumFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, chunk_len, n_heads, tune", [
                pytest.param(1, 2, 64, 4, False, marks=pytest.mark.smoke),
                pytest.param(2, 4, 64, 8, False, marks=pytest.mark.full),
                pytest.param(1, 2, 128, 4, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 16, False, marks=pytest.mark.full),
            ]),
        ]

class DaCumsumFwdTest(WorkloadBase):
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
