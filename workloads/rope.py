"""Workload definitions for the RoPE op family."""

import torch

from workloads.workload_base import WorkloadBase  # noqa: F401


class RopeWorkload(WorkloadBase):
    def __init__(self, variant: str, layout: str, batch: int, seq_len: int,
                 num_heads: int, head_dim: int, dtype: torch.dtype,
                 extra_kwargs: dict | None = None):
        self.variant = variant
        self.layout = layout
        self.batch = batch
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.extra_kwargs = extra_kwargs or {}

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """Generate only x; cos/sin are computed by the op internally."""
        if self.layout == "1d":
            x = torch.randn(self.seq_len, self.head_dim, device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.batch, self.seq_len, self.num_heads, self.head_dim,
                             device="cuda", dtype=self.dtype)
        return (x,)
