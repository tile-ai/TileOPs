import torch

from workloads.base import FixtureBase, WorkloadBase


class SsdDecodeFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest
        return [
            ("batch, n_heads, d_head, d_state, n_groups, dtype, tune", [
                pytest.param(
                    1, 4, 64, 16, 1, torch.float16, False, marks=pytest.mark.smoke,
                ),
                pytest.param(
                    2, 8, 64, 32, 2, torch.float16, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    1, 4, 64, 16, 1, torch.bfloat16, False, marks=pytest.mark.full,
                ),
                pytest.param(
                    2, 8, 128, 64, 4, torch.bfloat16, False, marks=pytest.mark.full,
                ),
            ]),
        ]

class SsdDecodeTest(WorkloadBase):
    def __init__(
        self,
        batch: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
    ):
        self.batch = batch
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype

    def gen_inputs(self):
        b, h, p, n, g = (
            self.batch, self.n_heads, self.d_head, self.d_state, self.n_groups,
        )
        # A <= 0 (negative decay), dt > 0 (post-softplus)
        A = -torch.rand(h, dtype=torch.float32, device="cuda")
        dt = torch.rand(b, h, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        x = torch.randn(b, h, p, dtype=self.dtype, device="cuda") * 0.1
        B_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        C_in = torch.randn(b, g, n, dtype=self.dtype, device="cuda") * 0.1
        state = torch.randn(b, h, p, n, dtype=torch.float32, device="cuda") * 0.1
        return A, dt, x, B_in, C_in, state
