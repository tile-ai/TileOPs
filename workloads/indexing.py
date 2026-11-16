import torch
import torch.nn.functional as F

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload


class EmbeddingWorkload(CallWorkload):
    """An ``EmbeddingFwdOp`` call: the indices and table its manifest call states."""

    def ref_program(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``F.embedding`` with the call's parameters."""
        params = self.arguments()
        params.pop("validate_inputs", None)
        return F.embedding(input, weight, **params)

    def verification(self, *inputs):
        # A row copy has no rounding.
        return Exact(rtol=0, atol=0)
