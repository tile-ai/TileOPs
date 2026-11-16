"""Workload definitions for the quantize ops of the quantization family.

Every reference computes in float32 on the upcast input. A scale is the dequantization
multiplier, ``x ~= q * scale``; an all-zero group gets scale 1.0; rounding is half to
even (``torch.round``).
"""

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import WorkloadBase

_BLOCK = 128


def _int8_scale(amax: torch.Tensor) -> torch.Tensor:
    """``amax / 127``, and 1.0 where ``amax`` is zero."""
    return torch.where(amax > 0, amax / 127, torch.ones_like(amax))


def _to_int8(xf: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.round(xf / scale).clamp(-127, 127).to(torch.int8)


def _pad_to_blocks(xf: torch.Tensor, rows: bool) -> torch.Tensor:
    """*xf* zero-padded along K, and along the rows too when *rows*, to whole blocks."""
    m, k = xf.shape
    pad_m = -m % _BLOCK if rows else 0
    return F.pad(xf, (0, -k % _BLOCK, 0, pad_m))


def int8_quant_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``INT8QuantPerTensorFwdOp``'s reference: one scale for the tensor."""
    _m, _k = x.shape
    xf = x.float()
    scale = _int8_scale(xf.abs().amax()).reshape(1)
    return _to_int8(xf, scale), scale


def int8_quant_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``INT8QuantPerChannelFwdOp``'s reference: one scale per row."""
    _n, _k = w.shape
    wf = w.float()
    scale = _int8_scale(wf.abs().amax(dim=1))
    return _to_int8(wf, scale[:, None]), scale


def int8_quant_per_block(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``INT8QuantPerTokenGroupFwdOp``'s reference: one scale per 128 elements of a row."""
    m, k = x.shape
    xf = x.float()
    blocks = _pad_to_blocks(xf, rows=False).view(m, -(-k // _BLOCK), _BLOCK)
    scale = _int8_scale(blocks.abs().amax(dim=-1))
    return _to_int8(xf, scale.repeat_interleave(_BLOCK, 1)[:, :k]), scale


def fp8_quant_per_block(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``FP8QuantPerBlockFwdOp``'s reference: one scale per 128x128 tile."""
    n, k = w.shape
    wf = w.float()
    padded = _pad_to_blocks(wf, rows=True)
    tiles = padded.view(padded.shape[0] // _BLOCK, _BLOCK, padded.shape[1] // _BLOCK, _BLOCK)
    amax = tiles.abs().amax(dim=(1, 3))
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.where(amax > 0, amax / fp8_max, torch.ones_like(amax))
    full = scale.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)[:n, :k]
    q = (wf / full).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale


def int4_quant_per_group(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSpeed asymmetric INT4: signed codes and float32 scale/offset pairs."""
    n, k = w.shape
    groups = w.float().view(-1, group_size)
    lo, hi = groups.amin(dim=-1), groups.amax(dim=-1)
    multiplier = torch.where(hi == lo, 1.0, 16.0 / (hi - lo))
    offset = (hi + lo) * 0.5
    q = ((groups - offset[:, None]) * multiplier[:, None]).round().clamp(-8, 7)
    q = q.to(torch.int8).view(n, k // 2, 2)
    packed = (q[..., 0] << 4) | (q[..., 1] & 15)
    return packed, torch.stack((1.0 / multiplier, offset), dim=-1)


def smooth_quant(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``SmoothQuantFwdOp``'s reference: divide by ``smooth``, then one scale per row."""
    _m, k = x.shape
    xs = x.float() / smooth.view(k)
    scale = _int8_scale(xs.abs().amax(dim=1))
    return _to_int8(xs, scale[:, None]), scale


def round_up_to_power_of_two(t: torch.Tensor) -> torch.Tensor:
    """``2 ** ceil(log2(t))`` of a positive float32 *t*, subnormals included."""
    mantissa, exponent = torch.frexp(t)
    return torch.ldexp(torch.ones_like(t), exponent - (mantissa == 0.5).to(exponent.dtype))


# Midpoints between adjacent e2m1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    """The e2m1 code of each float32 in ``[-6, 6]``, rounded to nearest, ties to the even code."""
    a = y.abs()
    magnitude = torch.zeros_like(a, dtype=torch.int32)
    for i, mid in enumerate(_E2M1_MIDPOINTS):
        # A value on a midpoint goes to the even code: down from an even one, up from an odd one.
        magnitude += (a > mid if i % 2 == 0 else a >= mid).to(torch.int32)
    return magnitude | (torch.signbit(y).to(torch.int32) << 3)


def _e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=codes.device)
    value = grid[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -value, value)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """The float32 values of ``uint8`` e2m1 pairs, the low nibble first, along a last axis
    twice as long."""
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return _e2m1_values(codes.to(torch.int32))


def fp4_quant_per_group(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """``FP4QuantPerGroupFwdOp``'s reference, DeepSeek-V4-Flash ``inference/kernel.py``
    ``fp4_act_quant``: per *block_size* elements along the last axis the scale is
    ``2 ** ceil(log2(max(amax, 6 * 2 ** -126) * fp32(1 / 6)))``, and each element is
    ``clamp(x / scale, -6, 6)`` rounded to e2m1.

    Returns ``(codes, scales, None)``, the ``uint8`` codes two per byte with the even element
    in the low nibble; with *inplace*, *x* holds ``codes * scale`` in its own dtype and
    ``(None, None, x)`` returns.
    """
    n = x.shape[-1]
    if block_size <= 0 or n % block_size or (n % 2 and not inplace):
        raise ValueError(f"the last axis {n} is not whole groups of {block_size} and bytes")
    groups = x.float().unflatten(-1, (n // block_size, block_size))
    amax = groups.abs().amax(-1, keepdim=True).clamp(min=6 * 2.0**-126)
    # A Python scalar multiplies in float32, as the kernel's reciprocal does.
    scale = round_up_to_power_of_two(amax * (1 / 6))
    codes = _e2m1_codes((groups / scale).clamp(-6, 6))
    if inplace:
        x.copy_((_e2m1_values(codes) * scale).flatten(-2))
        return None, None, x
    pairs = codes.flatten(-2).unflatten(-1, (n // 2, 2))
    packed = (pairs[..., 0] | (pairs[..., 1] << 4)).to(torch.uint8)
    return packed, scale.squeeze(-1).to(torch.float8_e8m0fnu), None


class _QuantizeWorkload(WorkloadBase):
    """One random ``[rows, cols]`` input of a quantize op."""

    # The manifest index naming the input's leading extent.
    _ROWS: ClassVar[str] = "M"

    def __init__(self, rows: int, cols: int, dtype: torch.dtype) -> None:
        self.rows = rows
        self.cols = cols
        self.dtype = dtype

    @classmethod
    def from_call(cls, call: Any) -> "_QuantizeWorkload":
        """The workload of one manifest call of the op this class is named for."""
        indices = call.indices
        return cls(indices[cls._ROWS], indices["K"], getattr(torch, indices["T"]))

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.rows, self.cols, dtype=self.dtype, device=run_device())
        return (x,)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        def validate(got, expected):
            codes, scales = got
            ref_codes, ref_scales = expected
            assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8)), (
                "quantized codes differ"
            )
            torch.testing.assert_close(scales, ref_scales, rtol=1e-6, atol=0, equal_nan=True)

        return Custom(validate, "exact quantized codes and scale rounding")


class INT8QuantPerTensorWorkload(_QuantizeWorkload):
    def ref_program(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int8_quant_per_tensor(x)


class INT8QuantPerChannelWorkload(_QuantizeWorkload):
    _ROWS = "N"

    def ref_program(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int8_quant_per_channel(w)


class INT8QuantPerBlockWorkload(_QuantizeWorkload):
    def ref_program(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int8_quant_per_block(x)


class FP8QuantPerBlockWorkload(_QuantizeWorkload):
    _ROWS = "N"

    def ref_program(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fp8_quant_per_block(w)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        def validate(got, expected):
            for actual, reference in zip(got, expected, strict=True):
                assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8)), (
                    "quantization bits differ"
                )

        return Custom(validate, "bitwise FP8 codes and block scales, including NaN and signed zero")


class INT4QuantPerGroupWorkload(_QuantizeWorkload):
    _ROWS = "N"

    def __init__(self, rows: int, cols: int, dtype: torch.dtype, group_size: int = 128) -> None:
        super().__init__(rows, cols, dtype)
        self.group_size = group_size

    @classmethod
    def from_call(cls, call: Any) -> "INT4QuantPerGroupWorkload":
        indices = call.indices
        return cls(indices["N"], indices["K"], getattr(torch, indices["T"]), indices["group_size"])

    def ref_program(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return int4_quant_per_group(w, self.group_size)

    def verification(self, *inputs):
        from workloads.numerics import Custom, zeroed_input

        def validate(got, expected):
            assert torch.equal(got[0], expected[0]), "packed INT4 codes differ"
            torch.testing.assert_close(got[1], expected[1], rtol=1e-6, atol=0)

        return Custom(
            validate,
            "exact packed INT4 codes and scale/offset rounding",
            controls=(zeroed_input(0, "weight-zeroed"),),
        )


class SmoothQuantWorkload(_QuantizeWorkload):
    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        (x,) = super().gen_inputs()
        # Smoothing factors are positive; values near zero would only blow up the input.
        smooth = torch.empty(self.cols, dtype=torch.float32, device=run_device()).uniform_(0.5, 2)
        return x, smooth

    def ref_program(
        self, x: torch.Tensor, smooth: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return smooth_quant(x, smooth)
