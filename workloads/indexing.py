"""Inputs and references of the indexing family."""

import torch

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload


class MaskedScatterWorkload(CallWorkload):
    """The inputs of one ``MaskedScatterFwdOp`` call.

    The manifest draws ``mask`` as random bools and ``source`` in the row's ``R`` shape.
    ``no_true`` makes every mask element false. ``true_byte`` stores each true element
    as that nonzero byte, which a bool tensor may hold. ``source_extra`` cuts ``source`` to the
    elements the true positions take, plus that many.
    """

    def __init__(
        self,
        call,
        device=None,
        *,
        no_true: bool = False,
        true_byte: int = 1,
        source_extra: "int | None" = None,
    ):
        super().__init__(call, device)
        self.no_true = no_true
        self.true_byte = true_byte
        self.source_extra = source_extra

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input, mask, source = super().gen_inputs()
        if self.no_true:
            mask.fill_(False)
        if self.source_extra is not None:
            taken = int(mask.expand(torch.broadcast_shapes(input.shape, mask.shape)).sum())
            source = source.flatten()[: taken + self.source_extra].clone()
        if self.true_byte != 1:
            mask = (mask.view(torch.uint8) * self.true_byte).view(torch.bool)
        return input, mask, source

    def ref_program(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        # A bool element holding any nonzero byte is true.
        return torch.masked_scatter(input, mask.view(torch.uint8) != 0, source)

    def torch_program(
        self, input: torch.Tensor, mask: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        return torch.masked_scatter(input, mask, source)

    def verification(self, *inputs):
        # Every output element is a copy of one input or source element.
        return Exact(rtol=0, atol=0)
