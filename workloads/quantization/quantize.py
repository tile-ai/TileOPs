"""Workload definitions for the quantize ops of the quantization family.

Every reference computes in float32 on the upcast input. A scale is the dequantization
multiplier, ``x ~= q * scale``; an all-zero group gets scale 1.0; rounding is half to
even (``torch.round``).
"""

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.gemm import repack_w4a16_weight
from workloads.workload_base import WorkloadBase

_BLOCK = 128
_FP8_MAX = 448.0


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
    """``INT8QuantPerBlockFwdOp``'s reference: one scale per 128 elements of a row."""
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
    scale = torch.where(amax > 0, amax / _FP8_MAX, torch.ones_like(amax))
    full = scale.repeat_interleave(_BLOCK, 0).repeat_interleave(_BLOCK, 1)[:n, :k]
    q = (wf / full).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def int4_quant_per_group(
    w: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``INT4QuantPerGroupFwdOp``'s reference: asymmetric, one scale and zero per group."""
    n, k = w.shape
    groups = w.float().view(n, k // group_size, group_size)
    # The range includes 0, so the zero point lies in [0, 15].
    lo = groups.amin(dim=-1).clamp_max(0)
    hi = groups.amax(dim=-1).clamp_min(0)
    floor = torch.finfo(w.dtype).tiny
    scale = torch.where(hi > lo, ((hi - lo) / 15).clamp_min(floor), torch.ones_like(hi))
    # Quantize against the scale the op returns.
    scale = scale.to(w.dtype).float()
    zero = torch.round(-lo / scale)
    q = (torch.round(groups / scale[..., None]) + zero[..., None]).clamp(0, 15)
    q = q.to(torch.uint8).view(n, k // 2, 2)
    # Row-major bytes, even K in the low nibble, reordered as GemmW4A16FwdOp.repack does.
    # That order is defined per 128-element K step, which the signature requires.
    packed = repack_w4a16_weight(q[..., 0] | (q[..., 1] << 4))
    return packed, scale.to(w.dtype), zero.to(torch.uint8)


def smooth_quant(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``SmoothQuantFwdOp``'s reference: divide by ``smooth``, then one scale per row."""
    _m, k = x.shape
    xs = x.float() / smooth.view(k)
    scale = _int8_scale(xs.abs().amax(dim=1))
    return _to_int8(xs, scale[:, None]), scale


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
        ix = call.ix
        return cls(ix[cls._ROWS], ix["K"], getattr(torch, ix["T"]))

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.rows, self.cols, dtype=self.dtype, device=run_device())
        return (x,)


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


class INT4QuantPerGroupWorkload(_QuantizeWorkload):
    _ROWS = "N"

    def __init__(self, rows: int, cols: int, dtype: torch.dtype, group_size: int = 128) -> None:
        super().__init__(rows, cols, dtype)
        self.group_size = group_size

    @classmethod
    def from_call(cls, call: Any) -> "INT4QuantPerGroupWorkload":
        ix = call.ix
        return cls(ix["N"], ix["K"], getattr(torch, ix["T"]), ix["group_size"])

    def ref_program(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return int4_quant_per_group(w, self.group_size)


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
