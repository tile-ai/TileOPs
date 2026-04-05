import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdChunkScanFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
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
