import torch
import torch.nn.functional as F

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


class EmbeddingWorkload(CallWorkload):
    """An ``EmbeddingFwdOp`` call: the indices and table its manifest call states."""

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The call's indices and table; the last index names the table's last row, and an
        integer table spans its dtype's range."""
        input, weight = super().gen_inputs()
        if input.numel():
            input.view(-1)[-1] = weight.shape[0] - 1
        if not weight.dtype.is_floating_point:
            info = torch.iinfo(weight.dtype)
            weight = torch.randint(
                info.min, info.max, weight.shape, dtype=weight.dtype, device=weight.device
            )
        return input, weight

    def ref_program(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The table rows the indices name, by indexing."""
        return weight[input.long()]

    def torch_program(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``F.embedding`` with the call's parameters."""
        params = self.arguments()
        params.pop("validate_inputs", None)
        return F.embedding(input, weight, **params)

    def verification(self, *inputs):
        # A row copy has no rounding.
        return Exact(rtol=0, atol=0)


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
