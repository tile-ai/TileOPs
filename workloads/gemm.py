"""Workloads for GEMM, batched matmul and grouped GEMM."""

from dataclasses import dataclass
from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128


def matmul_fp32_reduction(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``torch.matmul`` with any split-K partials reduced in fp32, as ``GemmFwdOp`` accumulates.

    torch lets cuBLAS reduce split-K partials in the 16-bit input dtype by default; on
    256x21843x1024 bf16 that put 23% of outputs more than one ulp off the exact product.
    """
    flags = torch.backends.cuda.matmul
    saved = (
        flags.allow_fp16_reduced_precision_reduction,
        flags.allow_bf16_reduced_precision_reduction,
    )
    flags.allow_fp16_reduced_precision_reduction = False
    flags.allow_bf16_reduced_precision_reduction = False
    try:
        return torch.matmul(a, b)
    finally:
        (
            flags.allow_fp16_reduced_precision_reduction,
            flags.allow_bf16_reduced_precision_reduction,
        ) = saved


class GemmWorkload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        trans_a: bool = False,
        trans_b: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    @classmethod
    def from_call(cls, call: Any) -> "GemmWorkload":
        """The workload of one manifest call of ``GemmFwdOp``."""
        indices = call.indices
        return cls(
            indices["M"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            indices["trans_a"],
            indices["trans_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device=run_device(), dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return matmul_fp32_reduction(a, b)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        tol = 1e-3 if self.dtype == torch.float16 else 1.6e-2
        gemv = not self.trans_a and (
            (self.m == 1 and self.trans_b) or (self.n == 1 and not self.trans_b)
        )
        atol = tol * max(1.0, self.k / 2048) if gemv else tol
        return Exact(atol=atol, rtol=tol)


class GemmFP8Workload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        scale_mode: str,
        out_dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.scale_mode = scale_mode
        self.out_dtype = out_dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "GemmFP8Workload":
        """The workload of one manifest call of ``GemmFP8FwdOp``."""
        indices = call.indices
        if tuple(indices["SA"]) == (1, 1):
            scale_mode = "per_tensor"
        elif tuple(indices["SB"]) == (indices["N"], -(-indices["K"] // 128)):
            scale_mode = "block128"
        else:
            scale_mode = "block128x128"
        return cls(
            indices["M"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            scale_mode,
            out_dtype=getattr(torch, indices["out_dtype"]),
            bias=call.present("bias"),
        )

    def _scale_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.scale_mode in ("per_tensor", "tensor"):
            return (1, 1), (1, 1)
        if self.scale_mode == "block128":
            if self.k % 128 != 0:
                raise ValueError("block128 FP8 workloads require k divisible by 128")
            return (self.m, self.k // 128), (self.n, self.k // 128)
        if self.scale_mode == "block128x128":
            if self.k % 128 != 0:
                raise ValueError("block128x128 FP8 workloads require k divisible by 128")
            return (self.m, self.k // 128), (-(-self.n // 128), self.k // 128)
        raise ValueError(f"unknown FP8 GEMM scale_mode {self.scale_mode!r}")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (torch.randn(self.m, self.k, device=run_device()) * 0.25).to(self.dtype).contiguous()
        b = (torch.randn(self.n, self.k, device=run_device()) * 0.25).to(self.dtype).contiguous()
        scale_a_shape, scale_b_shape = self._scale_shapes()
        scale_a = (
            0.5 + torch.rand(*scale_a_shape, device=run_device(), dtype=torch.float32)
        ).contiguous()
        scale_b = (
            0.5 + torch.rand(*scale_b_shape, device=run_device(), dtype=torch.float32)
        ).contiguous()
        if self.bias:
            bias = torch.randn(self.n, device=run_device(), dtype=self.out_dtype)
            return a, b, scale_a, scale_b, bias
        return a, b, scale_a, scale_b

    def _expand_scale(self, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """One scale per element of the ``[rows, cols]`` operand it multiplies.

        A scale grid is per tensor, per 1x128 block along ``cols``, or per 128x128 block.
        """
        if tuple(scale.shape) == (1, 1):
            return scale.expand(rows, cols)
        scale_cols = (cols + 127) // 128
        if tuple(scale.shape) == (rows, scale_cols):
            return scale.repeat_interleave(128, dim=1)[:, :cols]
        if tuple(scale.shape) == ((rows + 127) // 128, scale_cols):
            return scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
        raise ValueError(f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(scale_a, self.m, self.k)
        b_f = b.float() * self._expand_scale(scale_b, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)

    def verification(self, *inputs):
        return fp8_matmul_verification(self.k)


class GemmINT8W8A8Workload(WorkloadBase):
    """INT8 activations and weights, each with one scale per tensor or one per output row.

    The operands are quantized from float matrices by the library's own INT8 references:
    the activation per token (one scale per row) or per tensor, the weight per channel or
    per tensor.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        per_token: bool,
        per_channel: bool,
        out_dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.per_token = per_token
        self.per_channel = per_channel
        self.out_dtype = out_dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "GemmINT8W8A8Workload":
        """The workload of one manifest call of ``GemmINT8W8A8FwdOp``."""
        indices = call.indices
        m, n = indices["M"], indices["N"]
        return cls(
            m,
            n,
            indices["K"],
            per_token=m > 1 and tuple(indices["SA"]) == (m, 1),
            per_channel=n > 1 and tuple(indices["SB"]) == (1, n),
            out_dtype=getattr(torch, indices["out_dtype"]),
            bias=call.present("bias"),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        from workloads.quantization.quantize import int8_quant_per_channel, int8_quant_per_tensor

        x = torch.randn(self.m, self.k, device=run_device())
        w = torch.randn(self.n, self.k, device=run_device())
        if self.per_token:
            a, scale_a = int8_quant_per_channel(x)
            scale_a = scale_a.reshape(self.m, 1)
        else:
            a, scale_a = int8_quant_per_tensor(x)
            scale_a = scale_a.reshape(1, 1)
        if self.per_channel:
            b, scale_b = int8_quant_per_channel(w)
            scale_b = scale_b.reshape(1, self.n)
        else:
            b, scale_b = int8_quant_per_tensor(w)
            scale_b = scale_b.reshape(1, 1)
        inputs = (a.contiguous(), b.contiguous(), scale_a.contiguous(), scale_b.contiguous())
        if self.bias:
            return (*inputs, torch.randn(self.n, device=run_device(), dtype=self.out_dtype))
        return inputs

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The exact int32 product, then the fp32 epilogue and one rounding to ``out_dtype``.

        Every partial sum is an integer below ``127 * 127 * K``, exact in float64.
        """
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        acc = torch.matmul(a.double(), b.double().T).float()
        out = acc * scale_a * scale_b
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)

    def verification(self, *inputs):
        from workloads.numerics import Exact, zeroed_input

        # The product is exact on both sides; only the fp32 order of the two scale multiplies
        # and the bias add may differ, which moves a result across at most one rounding
        # boundary of the output: one ulp, 2^-7 of the value in bfloat16, 2^-10 in float16.
        rtol = 2**-7 if self.out_dtype == torch.bfloat16 else 2**-10
        return Exact(atol=1e-5, rtol=rtol, controls=(zeroed_input(0, "left-operand-zeroed"),))


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = W4A16_GROUP_SIZE,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine group-wise quantize and pack a logical ``[N, K]`` weight tensor.

    Args:
        weight: Logical weight, $[N \\times K]$.
        group_size: How many K values one scale and zero point cover.
        scale_dtype: Storage dtype of the scale, which follows the activation
            dtype an INT4 checkpoint is served at.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1).clamp_max(0)
    group_max = grouped.amax(dim=-1).clamp_min(0)
    scale = (
        ((group_max - group_min) / 15.0).to(scale_dtype).clamp_min(torch.finfo(scale_dtype).tiny)
    )
    scale_f = scale.float()
    zero = torch.round(-group_min / scale_f).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale_f.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )

    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = (
        (quantized.float() - zero.float().unsqueeze(-1)) * scale_f.unsqueeze(-1)
    ).reshape(n, k)
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


def repack_w4a16_weight(packed: torch.Tensor) -> torch.Tensor:
    """Reorder a row-major ``[N, K/2]`` uint8 weight for W4A16 GEMM.

    Args:
        packed: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``.

    Returns:
        A contiguous tensor with the same shape and dtype in the prepacked layout.
    """
    n, kp = packed.shape
    step = 64
    lanes = 4
    if kp % step:
        raise ValueError(f"K/2={kp} must be a multiple of {step}")
    n_steps = kp // step
    words_per_lane = step // (4 * lanes)
    grouped = packed.view(n, n_steps, step).to(torch.int32)
    low, high = grouped & 0xF, (grouped >> 4) & 0xF
    out = torch.zeros(n, n_steps, step // 4, dtype=torch.int32, device=packed.device)
    for lane in range(lanes):
        for word in range(words_per_lane):
            acc = torch.zeros(n, n_steps, dtype=torch.int32, device=packed.device)
            for pair in range(4):
                src = 16 * word + 4 * pair + lane
                acc = acc | (low[:, :, src] << (4 * pair)) | (high[:, :, src] << (4 * pair + 16))
            out[:, :, lane * words_per_lane + word] = acc
    return out.reshape(n, kp // 4).view(torch.uint8).reshape(n, kp).contiguous()


def unrepack_w4a16_weight(prepacked: torch.Tensor) -> torch.Tensor:
    """Invert :func:`repack_w4a16_weight`: the row-major ``[N, K/2]`` packing of a prepacked weight.

    Args:
        prepacked: Prepacked weights, ``[N, K/2]``, ``torch.uint8``.

    Returns:
        The row-major packed weights, two INT4 per byte, even K in the low nibble.
    """
    n, kp = prepacked.shape
    step = 64
    lanes = 4
    n_steps = kp // step
    words_per_lane = step // (4 * lanes)
    words = prepacked.contiguous().view(torch.int32).reshape(n, n_steps, step // 4)
    low = torch.zeros(n, n_steps, step, dtype=torch.int32, device=prepacked.device)
    high = torch.zeros_like(low)
    for lane in range(lanes):
        for word in range(words_per_lane):
            acc = words[:, :, lane * words_per_lane + word]
            for pair in range(4):
                src = 16 * word + 4 * pair + lane
                low[:, :, src] = (acc >> (4 * pair)) & 0xF
                high[:, :, src] = (acc >> (4 * pair + 16)) & 0xF
    return (low | (high << 4)).to(torch.uint8).reshape(n, kp)


def dequantize_w4a16_weight(
    prepacked: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """The logical ``[N, K]`` weight a prepacked INT4 weight and its group metadata encode.

    Weight ``(n, k)`` is ``(q - zero[n, g]) * scale[n, g]`` in float32, ``g`` the group of
    ``k`` and ``q`` its INT4 value.
    """
    packed = unrepack_w4a16_weight(prepacked)
    n, kp = packed.shape
    q = torch.empty(n, 2 * kp, dtype=torch.float32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).float()
    q[:, 1::2] = (packed >> 4).float()
    group_size = 2 * kp // scale.shape[1]
    zero_f = zero.float().repeat_interleave(group_size, dim=1)
    scale_f = scale.float().repeat_interleave(group_size, dim=1)
    return (q - zero_f) * scale_f


class GemmW4A16Workload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        group_size: int = W4A16_GROUP_SIZE,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self._row_major_weight: torch.Tensor | None = None

    @classmethod
    def from_call(cls, call: Any) -> "GemmW4A16Workload":
        """The workload of one manifest call of ``GemmW4A16FwdOp``."""
        indices = call.indices
        return cls(
            indices["M"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            indices["group_size"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return activation, prepacked weight, scale, and zero point."""
        activation = torch.randn(self.m, self.k, device=run_device(), dtype=self.dtype)
        source_weight = torch.randn(self.n, self.k, device=run_device(), dtype=torch.float32) * 0.25
        packed, scale, zero, _ = quantize_weight_int4(
            source_weight, group_size=self.group_size, scale_dtype=self.dtype
        )
        self._row_major_weight = packed
        return activation, repack_w4a16_weight(packed), scale, zero

    @property
    def row_major_weight(self) -> torch.Tensor:
        """The same weight before the repack, which a baseline reorders its own way."""
        if self._row_major_weight is None:
            raise RuntimeError("row_major_weight is available after gen_inputs()")
        return self._row_major_weight

    def ref_program(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        weight = dequantize_w4a16_weight(packed_weight, weight_scale, weight_zero)
        return torch.matmul(activation, weight.to(activation.dtype).T)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=7e-2, rtol=5e-2)


@dataclass(frozen=True)
class _W4A8Packing:
    """Extents of the W4A8 weight order the reference repack and the input sampler share.

    Attributes:
        bits: Bits of one signed INT4 value; two share a uint8 byte.
        unit_rows: Rows of one 512-byte unit.
        block_rows: Rows of one tile, the SM90 FP8 WGMMA M extent.
        group_k: K of one tile.
        warp_lanes: Lanes of one warp.
    """

    bits: int = torch.iinfo(torch.uint8).bits // 2
    unit_rows: int = 16
    block_rows: int = 64
    group_k: int = 128
    warp_lanes: int = 32


# The reference repack and the input sampler must encode the same signed INT4 order.
_W4A8_PACKING = _W4A8Packing()


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """The ``[N, K]`` INT4 values of a row-major ``[N, K // 2]`` uint8 packing.

    Byte ``(n, j)`` holds ``k = 2j`` in its low nibble and ``k = 2j + 1`` in its high nibble,
    each a signed two's-complement value in ``[-8, 7]``.
    """
    bits = _W4A8_PACKING.bits
    sign, mask = 1 << (bits - 1), (1 << bits) - 1
    nibbles = torch.stack((packed & mask, packed >> bits), dim=-1).to(torch.int16)
    return ((nibbles ^ sign) - sign).reshape(packed.shape[0], 2 * packed.shape[1])


def _w4a8_row_blocks(n: int, rows: int) -> list[tuple[int, int]]:
    """``(first row, extent)`` of each ``rows``-wide block, with an unpadded final block."""
    blocks = [(start, rows) for start in range(0, n - n % rows, rows)]
    if n % rows:
        blocks.append((n - n % rows, n % rows))
    return blocks


def repack_w4a8_weight(packed: torch.Tensor) -> torch.Tensor:
    """Reorder and re-encode a row-major ``[N, K/2]`` signed INT4 weight for W4A8 GEMM.

    Each INT4 ``q`` is stored as code ``q - 1`` for ``q`` in ``[1, 7]``, ``7`` for ``q = 0``
    and ``8 | (-q - 1)`` for ``q < 0``. Row ``16w + 8hr + r`` and K ``128g + 64h + 32p + 16s
    + 4t + i`` of a 64-row block go to nibble ``4s + i`` of word ``2p + hr`` of lane
    ``4r + t`` of the 512-byte unit ``(g, h, w)``; units are ordered ``[g][h][w]`` inside a
    block and blocks follow N. A last block of ``N % 64`` rows holds ``(N % 64) / 16`` units
    per ``(g, h)`` and is stored without padding.

    Args:
        packed: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``, even K in the low
            nibble, two's complement; positive ``N % 16 == 0`` and positive ``K % 128 == 0``.

    Returns:
        A contiguous ``[N, K/2]`` ``torch.uint8`` tensor in the prepacked order.
    """
    n, k = packed.shape[0], 2 * packed.shape[1]
    if n <= 0 or k <= 0:
        raise ValueError(f"N and K must be positive, got N={n}, K={k}")
    layout = _W4A8_PACKING
    if n % layout.unit_rows or k % layout.group_k:
        raise ValueError(
            f"N={n} must be a multiple of {layout.unit_rows} and K={k} of {layout.group_k}"
        )
    sign = 1 << (layout.bits - 1)
    q = unpack_signed_int4(packed)
    code = torch.where(q > 0, q - 1, torch.where(q == 0, sign - 1, sign | (-q - 1)))
    row_pair = layout.unit_rows // 2
    # Each lane's four FP8 values occupy one 32-bit fragment register.
    fragment_values = torch.int32.itemsize
    parts = []
    for start, rows in _w4a8_row_blocks(n, layout.block_rows):
        # Source axes w, hr, r, g, h, p, s, t, i; target g, h, w, r, t, p, hr, s, i.
        block = code[start : start + rows].reshape(
            rows // layout.unit_rows,
            2,
            row_pair,
            k // layout.group_k,
            2,
            2,
            2,
            layout.warp_lanes // row_pair,
            fragment_values,
        )
        nibbles = block.permute(3, 4, 0, 2, 7, 5, 1, 6, 8).reshape(-1, 2)
        parts.append((nibbles[:, 0] | (nibbles[:, 1] << layout.bits)).to(torch.uint8))
    return torch.cat(parts).reshape(n, k // 2)


def _w4a8_weight(packed: torch.Tensor, group_scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """The FP32 value of each INT4 weight times its group scale, rounded to float8_e4m3fn."""
    weight = unpack_signed_int4(packed).float() * group_scale.float().repeat_interleave(
        group_size, dim=1
    )
    return weight.to(torch.float8_e4m3fn).float()


def gemm_w4a8_reference(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_group_scale: torch.Tensor,
    weight_channel_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """The W4A8 semantic product with an explicit weight-group size.

    Args:
        activation: FP8 e4m3 activations, ``[M, K]``.
        packed_weight: Row-major signed INT4 pairs, ``[N, K/2]``, even K in the low nibble.
        weight_group_scale: FP8 e4m3 group scales, ``[N, K/group_size]``.
        weight_channel_scale: FP32 channel scales, ``[N]``.
        activation_scale: FP32 token scales, ``[M]``.
        group_size: Positive number of K values sharing one weight group scale.

    Returns:
        The FP32 contraction after nearest-even FP8 weight rounding, token-first FP32
        scaling and nearest-even BF16 output rounding, ``[M, N]``.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    weight = _w4a8_weight(packed_weight, weight_group_scale, group_size)
    out = torch.matmul(activation.float(), weight.T)
    out = out * activation_scale[:, None] * weight_channel_scale[None, :]
    return out.to(torch.bfloat16)


class GemmW4A8Workload(WorkloadBase):
    """FP8 activations times signed INT4 weights with a group, a channel and a token scale.

    The weight is generated in the row-major packing :func:`unpack_signed_int4` reads. The
    reference dequantizes it to ``q * weight_group_scale`` rounded to float8_e4m3fn, nearest
    even, contracts in fp32, and then scales by the token and channel scales.

    Numerical acceptance requires finite activations, rounded FP8 weights and finite channel
    and token scales. Every output is within ``5e-5 * K + 2e-2 * |r|`` of ``r``, the FP32
    reference with token-first scaling and BF16 rounding, and the whole output's squared
    error is at most ``5e-5`` of its and ``r``'s combined energy. Where a zero channel or
    token scale makes ``r`` zero, the output is exactly zero. Infinities and NaNs appear
    where ``r`` has them; NaN payloads and the sign of zero are unspecified.
    """

    def __init__(self, m: int, n: int, k: int, group_size: int) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.group_size = group_size

    @classmethod
    def from_call(cls, call: Any) -> "GemmW4A8Workload":
        """The workload of one manifest call of ``GemmW4A8FwdOp``."""
        indices = call.indices
        return cls(indices["M"], indices["N"], indices["K"], indices["group_size"])

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return activation, packed weight, group scale, channel scale and token scale.

        Activations are normal samples scaled by 0.25 before FP8 rounding. Signed INT4
        values cover [-8, 7], group scales are sampled from [0.03, 0.08) before FP8 rounding,
        and channel and token scales are sampled from [0.5, 1.5).
        """
        device = run_device()
        activation = (torch.randn(self.m, self.k, device=device) * 0.25).to(torch.float8_e4m3fn)
        bits = _W4A8_PACKING.bits
        sign, mask = 1 << (bits - 1), (1 << bits) - 1
        q = torch.randint(-sign, sign, (self.n, self.k), device=device, dtype=torch.int16)
        packed = ((q[:, 0::2] & mask) | ((q[:, 1::2] & mask) << bits)).to(torch.uint8)
        groups = self.k // self.group_size
        group_scale = (0.03 + 0.05 * torch.rand(self.n, groups, device=device)).to(
            torch.float8_e4m3fn
        )
        scale_lower = 0.5
        channel_scale = scale_lower + torch.rand(self.n, device=device)
        token_scale = scale_lower + torch.rand(self.m, device=device)
        return activation, packed.contiguous(), group_scale, channel_scale, token_scale

    def ref_program(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        return gemm_w4a8_reference(
            activation,
            packed_weight,
            weight_group_scale,
            weight_channel_scale,
            activation_scale,
            group_size=self.group_size,
        )

    def verification(self, *inputs):
        from workloads.numerics import (
            Custom,
            NegativeControl,
            assert_normalized_error,
        )

        activation, packed, group, channel, token = inputs
        weight = _w4a8_weight(packed, group, self.group_size)
        assert activation.float().isfinite().all() and weight.isfinite().all(), (
            "The numerical contract requires finite activations and rounded FP8 weights."
        )
        assert channel.isfinite().all() and token.isfinite().all(), (
            "The numerical contract requires finite channel and token scales."
        )
        # The W4A8 accuracy contract's absolute error per K term, relative error and
        # combined-energy fraction.
        absolute, relative, normalized = 5e-5 * self.k, 2e-2, 5e-5
        zero_scale = (channel[None, :] == 0) | (token[:, None] == 0)

        def validate(got, expected):
            torch.testing.assert_close(
                got,
                expected,
                atol=absolute,
                rtol=relative,
                equal_nan=True,
                msg="Every output retains the absolute and relative W4A8 accuracy bound.",
            )
            try:
                assert_normalized_error(got, expected, normalized)
            except AssertionError as error:
                raise AssertionError(
                    "The complete output retains the W4A8 combined-energy accuracy bound."
                ) from error
            exact_zero = zero_scale & (expected == 0)
            assert (got[exact_zero] == 0).all(), (
                "A zero channel or token scale gives an exactly zero output."
            )

        def invalid_output_class(reference, inputs):
            result = reference(*inputs).clone()
            first = result.reshape(-1)
            first[0] = float("inf") if first[0].isnan() else float("nan")
            return result

        return Custom(
            validator=validate,
            reason="per-element and energy bounds, with exact zeros under a zero scale",
            controls=(NegativeControl("output-class-changed", invalid_output_class),),
        )


class GemmW4A8MixedSignWorkload(GemmW4A8Workload):
    """:class:`GemmW4A8Workload` with the sign of each group scale drawn at random.

    The signature admits a group scale of either sign; the magnitudes, and so the
    verification, are :class:`GemmW4A8Workload`'s.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        activation, packed, group_scale, channel_scale, token_scale = super().gen_inputs()
        negative = torch.rand(group_scale.shape, device=group_scale.device) < 0.5
        signed = torch.where(negative, -group_scale.float(), group_scale.float())
        return activation, packed, signed.to(torch.float8_e4m3fn), channel_scale, token_scale


class BmmWorkload(WorkloadBase):
    """Workload for batched matmul: a=[B,M,K], b=[B,K,N] -> d=[B,M,N]."""

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

    @classmethod
    def from_call(cls, call: Any) -> "BmmWorkload":
        """The workload of one manifest call of ``BmmFwdOp``."""
        indices = call.indices
        return cls(
            indices["B"], indices["M"], indices["N"], indices["K"], getattr(torch, indices["T"])
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device=run_device(), dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BmmFP8Workload(WorkloadBase):
    """Workload for batched FP8 GEMM.

    ``a`` is ``[B, M, K]``; ``b`` is a contiguous ``[B, K, N]``, or ``[B, N, K]`` under
    ``trans_b``; ``scale_a`` / ``scale_b`` are rank-0 fp32 scalars in ``[0.5, 1.5)``.
    """

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype = torch.bfloat16,
        trans_b: bool = False,
    ) -> None:
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.trans_b = trans_b

    @classmethod
    def from_call(cls, call: Any) -> "BmmFP8Workload":
        """The workload of one manifest call of ``BmmFP8FwdOp``."""
        indices = call.indices
        return cls(
            indices["B"],
            indices["M"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            out_dtype=getattr(torch, indices["out_dtype"]),
            trans_b=indices["trans_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        init_scale = 0.25  # Keep FP8 products in range for this fixture.
        a = (
            (torch.randn(self.batch, self.m, self.k, device=run_device()) * init_scale)
            .to(self.dtype)
            .contiguous()
        )
        b = (
            (torch.randn(self.batch, self.k, self.n, device=run_device()) * init_scale)
            .to(self.dtype)
            .contiguous()
        )
        if self.trans_b:
            b = b.transpose(-2, -1).contiguous()
        scale_a = (0.5 + torch.rand((), device=run_device(), dtype=torch.float32)).contiguous()
        scale_b = (0.5 + torch.rand((), device=run_device(), dtype=torch.float32)).contiguous()
        return a, b, scale_a, scale_b

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs
        if self.trans_b:
            b = b.transpose(-2, -1)
        a_f = a.float() * scale_a
        b_f = b.float() * scale_b
        out = torch.bmm(a_f, b_f)
        return out.to(self.out_dtype)

    def verification(self, *inputs):
        return fp8_matmul_verification(self.k)


def _generate_batch_sizes(batch_sum: int, batch_count: int):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes[i] += 1
    return batch_sizes


def _generate_offsets(batch_sizes_list):
    batch_offsets_list = [0]
    for size in batch_sizes_list[:-1]:
        batch_offsets_list.append(batch_offsets_list[-1] + size)
    return batch_offsets_list


class GroupedGemmWorkload(WorkloadBase):
    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        N: int,
        K: int,
        dtype: torch.dtype,
        transpose_a: bool,
        transpose_b: bool,
    ):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)

    @classmethod
    def from_call(cls, call: Any) -> "GroupedGemmWorkload":
        """The workload of one manifest call of ``GroupedGemmFwdOp``."""
        indices = call.indices
        return cls(
            indices["M"],
            indices["G"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            indices["transpose_a"],
            indices["transpose_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = run_device()
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        batch_offsets_list = _generate_offsets(batch_sizes_list)

        if not self.transpose_a:
            # NT / NN: A is (batch_sum, K)
            A = torch.randn(batch_sum, K, device=device, dtype=dtype)
            if self.transpose_b:
                # NT: B is (batch_count, N, K)
                B = torch.randn(batch_count, N, K, device=device, dtype=dtype)
            else:
                # NN: B is (batch_count, K, N)
                B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
        else:
            # TN / TT: A is (batch_sum, N)
            A = torch.randn(batch_sum, N, device=device, dtype=dtype)
            if self.transpose_b:
                # TT: B is (K, batch_sum)
                B = torch.randn(K, batch_sum, device=device, dtype=dtype)
            else:
                # TN: B is (batch_sum, K)
                B = torch.randn(batch_sum, K, device=device, dtype=dtype)

        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets

    def ref_program(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.transpose_a:
            # NT / NN: output is (batch_sum, N)
            if self.transpose_b:
                # NT: A @ B^T
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[1]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i].transpose(0, 1).contiguous())
                    start = end
            else:
                # NN: A @ B
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i])
                    start = end
        else:
            # TN / TT: output is (batch_count, N, K)
            total_batch = int(batch_sizes.sum().item())
            assert A.shape[0] == total_batch
            N = A.shape[1]
            batch_count = len(batch_sizes)

            if self.transpose_b:
                # TT: A^T @ B^T
                K = B.shape[0]
                assert B.shape[1] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(
                        A[start:end].transpose(0, 1), B[:, start:end].transpose(0, 1)
                    )
                    start = end
            else:
                # TN: A^T @ B
                K = B.shape[1]
                assert B.shape[0] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(A[start:end].transpose(0, 1), B[start:end])
                    start = end
        return output


def fp8_matmul_verification(k):
    """FP8 accumulation noise grows with the square root of the reduction length."""
    from workloads.numerics import Exact, zeroed_input

    return Exact(
        atol=2e-2 * max(1.0, k / 1024) ** 0.5,
        rtol=2e-2,
        controls=(zeroed_input(0, "left-operand-zeroed"),),
    )


class GemmW4A16BasisWorkload(GemmW4A16Workload):
    """A basis vector selects exactly one dequantized weight: no reduction error."""

    def __init__(self, k_index):
        super().__init__(1, 35, 384, torch.float16)
        self.k_index = k_index

    def gen_inputs(self):
        n, k = 35, 384
        rows = torch.arange(n)[:, None]
        quantized = torch.randint(0, 16, (n, k), generator=self.rng("packed-weight"))
        zero = ((3 * rows + torch.arange(k // 128)[None, :]) % 16).to(torch.uint8)
        scale = (
            0.03137 + torch.arange(n * (k // 128), dtype=torch.float32).reshape(n, -1) * 0.001147
        ).to(torch.float16)
        packed = (quantized[:, 0::2] | (quantized[:, 1::2] << 4)).to(torch.uint8)
        group = self.k_index // 128
        centered = quantized[:, self.k_index].float() - zero[:, group].float()
        self.expected = (centered * scale[:, group].float()).half()[None, :]
        activation = torch.zeros((1, k), device=run_device(), dtype=torch.float16)
        activation[0, self.k_index] = 1
        prepacked = repack_w4a16_weight(packed)
        return activation, prepacked.to(run_device()), scale.to(run_device()), zero.to(run_device())

    def ref_program(self, activation, *weights):
        return self.expected.to(activation.device)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=0, rtol=0)


def w4a16_partition_verification():
    """Compare K partitions of the same decoded weights, not a separate dequantizer.

    Only FP32 partial-sum order changes. Keep the tighter partition consistency
    bound separate from the GEMM reference's weight-narrowing error allowance.
    """
    from workloads.numerics import Exact

    return Exact(atol=1e-5, rtol=2e-3)
