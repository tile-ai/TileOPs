from typing import Any

import torch

from workloads.device import run_device
from workloads.quantization.quantize import _round_up_to_power_of_two
from workloads.workload_base import WorkloadBase


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
        """Per group: ``scale = max(amax, eps) / 448``, rounded up to a power of two under
        ``use_ue8m0``, and ``q = clamp(x / scale, -448, 448)``; with ``inplace``, ``x`` becomes
        ``q * scale``. The result fills ``(q, scale, None)`` or ``(None, None, x)``."""
        groups = x.float().unflatten(-1, (-1, self.group_size))
        amax_value = torch.abs(groups).amax(dim=-1, keepdim=True).clamp(min=self.eps)
        scale = amax_value / 448.0
        if self.use_ue8m0:
            scale = _round_up_to_power_of_two(scale)
        q = torch.clamp(groups / scale, min=-448.0, max=448.0).to(torch.float8_e4m3fn)
        if self.inplace:
            x.copy_((q.float() * scale).flatten(-2))
            return None, None, x
        return q.flatten(-2), scale.squeeze(dim=-1).to(self.scale_dtype), None

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_quantized

        if not self.inplace:
            return Custom(assert_quantized, "FP8 codes and scales")

        def validate(got, expected):
            assert got[0] is None and got[1] is None, "an in-place call returns no q or scale"
            torch.testing.assert_close(got[2], expected[2], rtol=0, atol=0)

        return Custom(validate, "x fake-quantized in place, bit for bit")
