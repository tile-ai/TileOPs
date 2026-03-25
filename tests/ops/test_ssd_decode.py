import pytest
import torch

from tests.test_base import FixtureBase, TestBase, allclose_compare
from tileops.ops.ssd_decode import SsdDecodeOp


def ssd_decode_ref(
    A: torch.Tensor,      # (H,)          float32
    dt: torch.Tensor,     # (B, H)        float32
    x: torch.Tensor,      # (B, H, P)     any dtype
    B_in: torch.Tensor,   # (B, G, N)     any dtype
    C_in: torch.Tensor,   # (B, G, N)     any dtype
    state: torch.Tensor,  # (B, H, P, N)  float32  -- updated in-place
) -> torch.Tensor:
    """
    PyTorch reference for ssd_decode.

    Returns:
      y_out: (B, H, P)  float32

    Semantics:
      g                = h // (n_heads // n_groups)
      dA[b, h]         = exp(dt[b, h] * A[h])
      state[b,h,p,n]  <- dA[b,h] * state[b,h,p,n]
                         + dt[b,h] * B_in[b,g,n] * x[b,h,p]   (in-place)
      y_out[b, h, p]   = sum_n  state[b, h, p, n] * C_in[b, g, n]
    """
    B, H = dt.shape
    G = B_in.shape[1]
    heads_per_group = H // G

    dA = torch.exp(dt.float() * A.float())          # (B, H)

    # Expand B/C from groups to heads: (B, H, N)
    head_idx = torch.arange(H, device=B_in.device) // heads_per_group
    B_heads = B_in.float()[:, head_idx, :]           # (B, H, N)
    C_heads = C_in.float()[:, head_idx, :]           # (B, H, N)

    # dBx[b, h, p, n] = dt[b,h] * B[b,h,n] * x[b,h,p]
    dBx = (
        dt.float()[:, :, None, None]
        * x.float()[:, :, :, None]
        * B_heads[:, :, None, :]
    )  # (B, H, P, N)

    # Update state in-place
    new_state = dA[:, :, None, None] * state.float() + dBx  # (B, H, P, N)
    state.copy_(new_state)

    # y_out[b, h, p] = sum_n state[b, h, p, n] * C[b, h, n]
    y_out = torch.einsum("bhpn,bhn->bhp", state.float(), C_heads)
    return y_out


class SsdDecodeFixture(FixtureBase):
    PARAMS = [
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


class SsdDecodeTest(TestBase):
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

    def ref_program(self, A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)


@SsdDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SsdDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SsdDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Run reference on a clone of state so the two runs start from the same point.
    state_ref = state.clone()
    y_ref = test.ref_program(A, dt, x, B_in, C_in, state_ref)

    # Run kernel; state is updated in-place.
    y_op = op(A, dt, x, B_in, C_in, state)

    atol = 1e-3 if dtype == torch.float16 else 1.6e-2
    rtol = 1e-3
    allclose_compare(y_op, y_ref, atol=atol, rtol=rtol)
    allclose_compare(state, state_ref, atol=atol, rtol=rtol)
