import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdChunkStateFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune, has_seq_idx", [
                pytest.param(
                    1, 2, 64, 4, 64, 32, 1, torch.float16, False, False, marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2, 4, 64, 8, 64, 64, 2, torch.float16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 2, 128, 4, 128, 32, 1, torch.bfloat16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 2, 64, 4, 64, 32, 2, torch.bfloat16, False, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 4, 64, 8, 64, 64, 2, torch.float16, False, True, marks=pytest.mark.full,
                ),
            ]),
        ]

class SsdChunkStateFwdTest(WorkloadBase):
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
        has_seq_idx: bool = False,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.has_seq_idx = has_seq_idx

    def gen_inputs(self):
        b, c, Q, h, p, n, g = (
            self.batch, self.num_chunks, self.chunk_len,
            self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        seq_len = c * Q
        x = torch.randn(b, seq_len, h, p, dtype=self.dtype, device="cuda") * 0.1
        Bmat = torch.randn(b, seq_len, g, n, dtype=self.dtype, device="cuda") * 0.1
        # dA_cumsum: monotonically non-increasing (negative values, cumsum of negatives)
        dA_cumsum = -torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda").cumsum(-1)
        dt = torch.rand(b, h, c, Q, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        seq_idx = None
        if self.has_seq_idx:
            # simulate two packed sequences per batch row, split at midpoint
            seq_idx = torch.zeros(b, seq_len, dtype=torch.int32, device="cuda")
            seq_idx[:, seq_len // 2:] = 1
        return x, Bmat, dt, dA_cumsum, seq_idx
