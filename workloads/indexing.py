"""Workloads of the indexing family."""

import torch
import torch.nn.functional as F

from workloads.numerics import Exact
from workloads.workload_base import CallWorkload


class EmbeddingCall(CallWorkload):
    """A manifest call of EmbeddingFwdOp, its table filled so that every row differs.

    ``strided`` hands the ids and the table as views of every second element of their last
    axis.
    """

    def __init__(self, call, strided=False):
        super().__init__(call)
        self.strided = strided

    def gen_inputs(self):
        input, weight = super().gen_inputs()
        # The last id names the table's last row, the one at the largest offset.
        input.view(-1)[-1:] = weight.shape[0] - 1
        if not weight.dtype.is_floating_point:
            # The default integer fill takes 8 values, so a row read from the wrong id would
            # often match; values over the dtype's whole range do not.
            info = torch.iinfo(weight.dtype)
            gen = self.rng("weight", device=weight.device)
            weight = torch.randint(
                info.min, info.max, weight.shape, generator=gen, device=weight.device
            ).to(weight.dtype)
        if self.strided:
            input = torch.stack([input, input], -1)[..., 0]
            weight = torch.stack([weight, weight], -1)[..., 0]
        return input, weight

    def ref_program(self, input, weight):
        """The rows ``input`` names, by advanced indexing."""
        return weight[input.long()]

    def torch_program(self, input, weight):
        return F.embedding(input, weight)

    def verification(self, *inputs):
        # A copy has no rounding.
        return Exact(rtol=0, atol=0)
