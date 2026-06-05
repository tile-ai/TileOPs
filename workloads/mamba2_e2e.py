"""End-to-end Mamba-2 SSD forward workload fixtures and test generators.

Covers model-scale configurations (130M–2.7B) and workload types
(latency / serving / throughput / long-context).
"""

import torch

from workloads.base import FixtureBase, WorkloadBase

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

class Mamba2FwdTest(WorkloadBase):
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
