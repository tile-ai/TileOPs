import torch

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload


class ConcatWorkload(CallWorkload):
    """A ``ConcatFwdOp`` call: the tensors its manifest call states, joined along ``dim``."""

    def ref_program(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """``torch.cat``, which converts every element to the promoted dtype."""
        return torch.cat(tensors, self.arguments()["dim"])

    def verification(self, *inputs):
        # A copy and an exact widening: every element equals the reference bit for bit.
        return Exact(rtol=0, atol=0)


class TransposedConcatWorkload(ConcatWorkload):
    """A ``ConcatFwdOp`` call on rank-2 elements that are transposed views of their storage."""

    def gen_inputs(self) -> tuple[list[torch.Tensor]]:
        (tensors,) = super().gen_inputs()
        return ([t.t().contiguous().t() for t in tensors],)


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
