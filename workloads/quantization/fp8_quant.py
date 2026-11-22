from typing import Any

import torch

from workloads.device import run_device
from workloads.quantization.quantize import (
    StridedInput,
    bitwise_verification,
    fill_with_ties,
    round_up_to_power_of_two,
)
from workloads.workload_base import WorkloadBase


def fp8_quant_per_group(
    x: torch.Tensor,
    group_size: int,
    eps: float,
    use_ue8m0: bool | None,
    scale_dtype: torch.dtype,
    inplace: bool,
):
    """Per ``group_size`` elements of the last axis: ``scale = max(amax, eps) / 448``, under
    ``use_ue8m0`` rounded up to a power of two no smaller than ``2 ** -126``, and
    ``q = clamp(x / scale, -448, 448)`` in e4m3. Returns ``(q, scale)``, or with ``inplace``
    ``x`` overwritten by ``q * scale``."""
    groups = x.float().unflatten(-1, (-1, group_size))
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 448.0
    if use_ue8m0:
        scale = round_up_to_power_of_two(scale.clamp(min=2.0**-126))
    q = torch.clamp(groups / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
    if inplace:
        return x.copy_((q.float() * scale).flatten(-2))
    return q.flatten(-2), scale.squeeze(dim=-1).to(scale_dtype)


class FP8QuantWorkload(WorkloadBase):
    """Group-wise FP8 quantization of a ``shape`` tensor, groups of ``group_size`` columns."""

    def __init__(
        self,
        shape: tuple[int, ...],
        group_size: int,
        in_dtype: torch.dtype,
        eps: float = 1e-10,
        out_q: bool = False,
        use_ue8m0: bool | None = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
    ):
        self.shape = tuple(shape)
        self.group_size = group_size
        self.in_dtype = in_dtype
        self.eps = eps
        self.out_q = out_q
        self.use_ue8m0 = use_ue8m0
        self.scale_dtype = scale_dtype
        self.inplace = inplace

    @classmethod
    def from_call(cls, call: Any) -> "FP8QuantWorkload":
        """The workload of one manifest call of ``FP8QuantPerTokenGroupFwdOp``."""
        shape, dtype = call.tensors["x"]
        params = call.params
        return cls(
            shape,
            params["group_size"],
            getattr(torch, dtype),
            params["eps"],
            call.specs["out_q"] is not None,
            params["use_ue8m0"],
            getattr(torch, params["scale_dtype"]),
            params["inplace"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, dtype=self.in_dtype, device=run_device())
        if not self.out_q:
            return (x,)
        return x, torch.empty(self.shape, dtype=torch.float8_e4m3fn, device=run_device())

    def ref_program(self, x: torch.Tensor, out_q: torch.Tensor | None = None) -> tuple:
        """:func:`fp8_quant_per_group`, filling ``(q, scale, None)`` or ``(None, None, x)``."""
        out = fp8_quant_per_group(
            x, self.group_size, self.eps, self.use_ue8m0, self.scale_dtype, self.inplace
        )
        return (None, None, out) if self.inplace else (*out, None)

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_quantized

        if not self.use_ue8m0:
            return Custom(assert_quantized, "FP8 codes and scales")

        # A power-of-two scale makes x / scale exact, so only the rounding to e4m3 remains.
        return bitwise_verification("UE8M0 codes, scales and fake-quantized values, bit for bit")


class FP8QuantViewWorkload(StridedInput, FP8QuantWorkload):
    """``FP8QuantWorkload`` on a strided ``x``."""


class FP8QuantTiesWorkload(FP8QuantWorkload):
    """The workload on rows of e4m3 rounding ties, each row scaled by its own power of two,
    and on all-zero rows, whose maximum is the ``eps`` floor."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, *rest = super().gen_inputs()
        # Values on the midpoint of two adjacent e4m3 codes under a scale of 1, which a group
        # whose maximum is 448 has, each beside the code it rounds to (the even one); signed
        # zeros; and values below half the smallest subnormal code.
        midpoints = (1.0625, 1.1875, 2.0**-10, 3 * 2.0**-10, 272.0, 304.0, 432.0)
        ties = (448.0, *midpoints, 0.0, -0.0, *(-m for m in midpoints), 2.0**-12)
        # Within float16's range: 448 * 2 ** 7 and 2 ** -12 * 2 ** -8 are both normal or zero.
        fill_with_ties(x, self.group_size, ties, 16)
        return (x, *rest)


class FP8QuantScaleFloorWorkload(FP8QuantWorkload):
    """The workload on rows scaled by ``2 ** -(100 + r % 40)``, row ``r``, so that group
    maxima times 1 / 448 run from normal float32 through its subnormals, and on all-zero
    rows: the UE8M0 scales reach the ``2 ** -126`` floor under a small ``eps``."""

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, *rest = super().gen_inputs()
        rows = x.view(-1, x.shape[-1])
        shift = -(100 + torch.arange(rows.shape[0], device=x.device) % 40)
        rows.copy_((rows.float() * torch.exp2(shift.float())[:, None]).to(x.dtype))
        rows[1::5] = 0
        return (x, *rest)
