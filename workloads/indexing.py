import torch

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload


class MaskedScatterWorkload(CallWorkload):
    """A ``MaskedScatterFwdOp`` call: the input, mask and source its manifest call states."""

    def ref_program(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """``torch.masked_scatter``."""
        return torch.masked_scatter(input, mask, source)

    def verification(self, *inputs):
        # Elements are moved, never rounded.
        return Exact(rtol=0, atol=0)
