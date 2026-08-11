"""End-to-end Mamba-2 SSD forward workload fixtures and input generators.

Covers model-scale configurations (130M–2.7B) and workload types
(latency / serving / throughput / long-context).
"""

import torch
import torch.nn.functional as F

from workloads.workload_base import FixtureBase, WorkloadBase

# ---------------------------------------------------------------------------
# Model configs (Mamba-2 paper Table 1 / mamba_ssm reference)
# ---------------------------------------------------------------------------
MAMBA2_MODELS = {
    # label: (n_heads, d_head, d_state, n_groups)
    "130m": (24,   64, 128, 1),
    "370m": (48,   64, 128, 1),
    "780m": (64,   64, 128, 1),
    "1.3b": (80,   64, 128, 1),
    "2.7b": (128,  64, 128, 1),
}


def mamba2_fwd_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    dt_bias: torch.Tensor | None,
    chunk_size: int,
    dt_softplus: bool,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch SSD factorization returning both FP32 outputs."""
    batch, seqlen, n_heads, d_head = x.shape
    d_state = B.shape[-1]
    n_groups = B.shape[2]
    heads_per_group = n_heads // n_groups
    num_chunks = seqlen // chunk_size

    dt_value = dt.float()
    if dt_bias is not None:
        dt_value = dt_value + dt_bias.float()
    if dt_softplus:
        dt_value = F.softplus(dt_value)
    dt_value = torch.clamp(dt_value, min=0.0)
    dt_chunked = dt_value.reshape(
        batch, num_chunks, chunk_size, n_heads
    ).permute(0, 3, 1, 2)
    dA_cumsum = (
        dt_chunked * A.float().view(1, n_heads, 1, 1)
    ).cumsum(dim=-1)

    B_chunked = B.float().reshape(
        batch, num_chunks, chunk_size, n_groups, d_state
    )
    C_chunked = C.float().reshape(
        batch, num_chunks, chunk_size, n_groups, d_state
    )
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", C_chunked, B_chunked)
    mask = torch.ones(
        chunk_size, chunk_size, device=x.device, dtype=torch.bool
    ).tril()
    cb = cb * mask.view(1, 1, 1, chunk_size, chunk_size)

    decay = torch.exp(dA_cumsum[..., -1:] - dA_cumsum)
    decay_chunked = decay.permute(0, 2, 3, 1)
    dt_by_chunk = dt_chunked.permute(0, 2, 3, 1)
    x_chunked = x.float().reshape(
        batch, num_chunks, chunk_size, n_heads, d_head
    )
    head_groups = torch.arange(n_heads, device=x.device) // heads_per_group
    B_heads = B_chunked[:, :, :, head_groups, :]
    weighted_x = x_chunked * (decay_chunked * dt_by_chunk).unsqueeze(-1)
    chunk_states = torch.einsum("bcqhp,bcqhn->bchpn", weighted_x, B_heads)

    exp_dA_chunk = torch.exp(dA_cumsum[..., -1])
    if initial_states is None:
        state = torch.zeros(
            batch, n_heads, d_head, d_state,
            device=x.device,
            dtype=torch.float32,
        )
    else:
        state = initial_states.float()
    prev_states = []
    for chunk_idx in range(num_chunks):
        prev_states.append(state.unsqueeze(1))
        scale = exp_dA_chunk[:, :, chunk_idx].view(batch, n_heads, 1, 1)
        state = scale * state + chunk_states[:, chunk_idx]
    prev_states_tensor = torch.cat(prev_states, dim=1)

    dA_by_chunk = dA_cumsum.permute(0, 2, 3, 1)
    C_heads = C_chunked[:, :, :, head_groups, :]
    y_history = torch.einsum(
        "bcqhn,bchpn->bcqhp", C_heads, prev_states_tensor
    )
    y_history = y_history * torch.exp(dA_by_chunk).unsqueeze(-1)

    decay_ls = torch.exp(
        dA_cumsum.unsqueeze(-1) - dA_cumsum.unsqueeze(-2)
    ).masked_fill(
        ~mask.view(1, 1, 1, chunk_size, chunk_size), 0.0
    ).permute(0, 2, 1, 3, 4)
    cb_heads = cb[:, :, head_groups, :, :]
    local_cb = (
        cb_heads
        * decay_ls
        * dt_by_chunk.permute(0, 1, 3, 2).unsqueeze(-2)
    )
    x_by_head = x_chunked.permute(0, 1, 3, 2, 4)
    y_intra = torch.einsum(
        "bchls,bchsp->bchlp", local_cb, x_by_head
    ).permute(0, 1, 3, 2, 4)

    y = (y_history + y_intra).reshape(batch, seqlen, n_heads, d_head)
    return y.float(), state.float()


def mamba2_direct_ref(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Literal token-ordered primary recurrence for small diagnostic cases."""
    batch, seqlen, n_heads, d_head = x.shape
    d_state = B.shape[-1]
    heads_per_group = n_heads // B.shape[2]
    head_groups = torch.arange(n_heads, device=x.device) // heads_per_group
    state = torch.zeros(
        batch, n_heads, d_head, d_state,
        device=x.device,
        dtype=torch.float32,
    )
    outputs = []
    delta = F.softplus(dt.float()).clamp(min=0.0)
    for token in range(seqlen):
        delta_t = delta[:, token]
        B_heads = B[:, token, head_groups].float()
        C_heads = C[:, token, head_groups].float()
        update = (
            delta_t[:, :, None, None]
            * x[:, token].float()[..., None]
            * B_heads[:, :, None, :]
        )
        decay = torch.exp(delta_t * A.float())[..., None, None]
        state = decay * state + update
        outputs.append(
            (state * C_heads[:, :, None, :]).sum(dim=-1).unsqueeze(1)
        )
    return torch.cat(outputs, dim=1), state

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

class Mamba2FwdFixture(FixtureBase):
    """pytest parametrize fixture for Mamba2FwdOp benchmarks."""

    @classmethod
    def get_params(cls):
        import pytest

        smoke_params = []
        full_params = []

        # Smoke: small configs to verify correctness quickly
        smoke_params += [
            pytest.param(
                1, 256, 4, 64, 128, 1, torch.bfloat16, 256, True, False,
                id="smoke-b1-s256-4h",
                marks=pytest.mark.smoke,
            ),
            pytest.param(
                2, 512, 8, 64, 128, 1, torch.bfloat16, 256, True, False,
                id="smoke-b2-s512-8h",
                marks=pytest.mark.smoke,
            ),
        ]

        # Full: model-scale workloads
        workloads = [
            # (batch, seqlen, label)
            (1,  2048,  "latency"),
            (8,  2048,  "serving"),
            (32, 2048,  "throughput"),
            (4,  32768, "long-ctx"),
        ]
        for model_label, (n_heads, d_head, d_state, n_groups) in MAMBA2_MODELS.items():
            for batch, seqlen, wl_label in workloads:
                full_params.append(
                    pytest.param(
                        batch, seqlen, n_heads, d_head, d_state, n_groups,
                        torch.bfloat16, 256, True, False,
                        id=f"full-{model_label}-{wl_label}",
                        marks=pytest.mark.full,
                    )
                )

        return [
            (
                "batch, seqlen, n_heads, d_head, d_state, n_groups, "
                "dtype, chunk_size, dt_softplus, tune",
                smoke_params + full_params,
            )
        ]


# ---------------------------------------------------------------------------
# WorkloadBase subclass — generates all required input tensors
# ---------------------------------------------------------------------------

class Mamba2FwdWorkload(WorkloadBase):
    """Input generator for the Mamba-2 SSD end-to-end forward pass.

    Generates tensors matching the interface of Mamba2FwdOp.forward and
    mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined.
    """

    def __init__(
        self,
        batch: int,
        seqlen: int,
        n_heads: int,
        d_head: int,
        d_state: int,
        n_groups: int,
        dtype: torch.dtype,
        chunk_size: int = 256,
        dt_softplus: bool = True,
    ):
        self.batch = batch
        self.seqlen = seqlen
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.n_groups = n_groups
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.dt_softplus = dt_softplus
        self.num_chunks = seqlen // chunk_size

    def gen_inputs(self):
        """Return (x, dt, A, B, C, dt_bias) on CUDA.

        Tensor shapes:
            x:       (batch, seqlen, n_heads, d_head)          dtype
            dt:      (batch, seqlen, n_heads)                   float32
            A:       (n_heads,)                                 float32  (≤ 0)
            B:       (batch, seqlen, n_groups, d_state)         dtype
            C:       (batch, seqlen, n_groups, d_state)         dtype
            dt_bias: (n_heads,)                                 float32
        """
        b   = self.batch
        S   = self.seqlen
        h   = self.n_heads
        p   = self.d_head
        n   = self.d_state
        g   = self.n_groups
        dev = "cuda"
        dt  = self.dtype

        x       = torch.randn(b, S, h, p, dtype=dt,            device=dev) * 0.1
        dt_raw  = torch.randn(b, S, h,    dtype=torch.float32, device=dev) * 0.5
        A       = -torch.rand(h,           dtype=torch.float32, device=dev)        # negative decay
        B       = torch.randn(b, S, g, n, dtype=dt,            device=dev) * 0.1
        C       = torch.randn(b, S, g, n, dtype=dt,            device=dev) * 0.1
        dt_bias = torch.randn(h,           dtype=torch.float32, device=dev) * 0.1

        return x, dt_raw, A, B, C, dt_bias

    def ref_program(self, x, dt, A, B, C, dt_bias):
        return mamba2_fwd_ref(
            x, dt, A, B, C, dt_bias, self.chunk_size, self.dt_softplus
        )[0]


class Mamba2PrimaryWorkload(Mamba2FwdWorkload):
    """The manifest contract: no bias or initial state, two FP32 outputs."""

    def gen_inputs(self):
        return super().gen_inputs()[:5]

    def ref_program(self, x, dt, A, B, C):
        return mamba2_fwd_ref(
            x, dt, A, B, C, None, self.chunk_size, self.dt_softplus
        )
