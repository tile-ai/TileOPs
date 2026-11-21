"""Workloads for GEMM, batched matmul and grouped GEMM."""

from collections.abc import Iterator
from contextlib import contextmanager
from itertools import accumulate
from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128


@contextmanager
def fp32_matmul_reduction() -> Iterator[None]:
    """cuBLAS reduces fp16 and bf16 GEMM partials in fp32 inside the block, as ``GEMMFwdOp``
    accumulates; the prior settings, the split-K flags included, are restored after."""
    flags = torch.backends.cuda.matmul
    names = ("allow_fp16_reduced_precision_reduction", "allow_bf16_reduced_precision_reduction")
    # Assigning a bool resets the split-K flag, so the pair is restored together.
    saved = [(getattr(flags, n), getattr(flags, f"{n}_split_k", None)) for n in names]
    for name in names:
        setattr(flags, name, False)
    try:
        yield
    finally:
        for name, (allow, split_k) in zip(names, saved, strict=True):
            setattr(flags, name, allow if split_k is None else (allow, split_k))


def matmul_fp32_reduction(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """``torch.matmul``, or ``torch.addmm`` with a bias, under :func:`fp32_matmul_reduction`."""
    with fp32_matmul_reduction():
        return torch.matmul(a, b) if bias is None else torch.addmm(bias, a, b)


def _summation_order_verification(dtype: torch.dtype, length: int):
    """Tolerances of a product rounded to *dtype* that sums *length* terms per output element.

    The kernel sums them in another order than the float32 reference; a cancelled output
    keeps only that order's error, which grows with *length*.

    A float32 output of 16-bit products keeps the tensor cores' accumulation error: each
    step adds its products to the float32 sum with truncation, so the error of standard
    normal operands grows as ``length ** 1.5`` (on H200, at most 1.5e-3 over 4096 terms
    and 7.0e-3 over 16384, as cuBLAS's 16-bit GEMM with a float32 output). The bound
    keeps three times that, and stays far below the error of rounding to bfloat16
    (1.0 over 4096 terms).
    """
    from workloads.numerics import Exact

    if dtype == torch.float32:
        return Exact(atol=max(1e-4, 2e-3 * (length / 2048) ** 1.5), rtol=1e-5)
    tol = 1e-3 if dtype == torch.float16 else 1.6e-2
    return Exact(atol=tol * max(1.0, length / 2048), rtol=tol)


class GEMMWorkload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        trans_a: bool = False,
        trans_b: bool = False,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.bias = bias
        self.out_dtype = out_dtype

    @classmethod
    def from_call(cls, call: Any) -> "GEMMWorkload":
        """The workload of one manifest call of ``GEMMFwdOp``."""
        indices = call.indices
        return cls(
            indices["M"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            indices["trans_a"],
            indices["trans_b"],
            bias=call.present("bias"),
            out_dtype=None
            if call.params["out_dtype"] is None
            else getattr(torch, call.params["out_dtype"]),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device=run_device(), dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device=run_device(), dtype=self.dtype)
        if self.bias:
            return a, b, torch.randn(self.n, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        out = torch.matmul(a.float(), b.float())
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype or a.dtype)

    def torch_program(
        self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        a, b = a.T if self.trans_a else a, b.T if self.trans_b else b
        if self.out_dtype is not None:
            # F.linear on float32-cast operands, as DeepSeek-V4-Flash computes router logits.
            out = torch.matmul(a.float(), b.float())
            return out if bias is None else out + bias.float()
        return matmul_fp32_reduction(a, b, bias)

    def verification(self, *inputs):
        return _summation_order_verification(self.out_dtype or self.dtype, self.k)


class GEMMFP8Workload(WorkloadBase):
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
    def from_call(cls, call: Any) -> "GEMMFP8Workload":
        """The workload of one manifest call of ``GEMMFP8FwdOp``."""
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
        if self.scale_mode == "per_tensor":
            return (1, 1), (1, 1)
        if self.scale_mode not in ("block128", "block128x128"):
            raise ValueError(f"unknown FP8 GEMM scale_mode {self.scale_mode!r}")
        if self.k % 128 != 0:
            raise ValueError(f"{self.scale_mode} FP8 workloads require k divisible by 128")
        b_rows = self.n if self.scale_mode == "block128" else -(-self.n // 128)
        return (self.m, self.k // 128), (b_rows, self.k // 128)

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (torch.randn(self.m, self.k, device=run_device()) * 0.25).to(self.dtype)
        b = (torch.randn(self.n, self.k, device=run_device()) * 0.25).to(self.dtype)
        scale_a_shape, scale_b_shape = self._scale_shapes()
        a_scale = 0.5 + torch.rand(*scale_a_shape, device=run_device(), dtype=torch.float32)
        b_scale = 0.5 + torch.rand(*scale_b_shape, device=run_device(), dtype=torch.float32)
        if self.bias:
            bias = torch.randn(self.n, device=run_device(), dtype=self.out_dtype)
            return a, b, a_scale, b_scale, bias
        return a, b, a_scale, b_scale

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
        a, b, a_scale, b_scale = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(a_scale, self.m, self.k)
        b_f = b.float() * self._expand_scale(b_scale, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)

    def verification(self, *inputs):
        return fp8_matmul_verification(self.k)


class GEMMINT8W8A8Workload(WorkloadBase):
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
    def from_call(cls, call: Any) -> "GEMMINT8W8A8Workload":
        """The workload of one manifest call of ``GEMMINT8W8A8FwdOp``."""
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
            a, a_scale = int8_quant_per_channel(x)
            a_scale = a_scale.reshape(self.m, 1)
        else:
            a, a_scale = int8_quant_per_tensor(x)
            a_scale = a_scale.reshape(1, 1)
        if self.per_channel:
            b, b_scale = int8_quant_per_channel(w)
            b_scale = b_scale.reshape(1, self.n)
        else:
            b, b_scale = int8_quant_per_tensor(w)
            b_scale = b_scale.reshape(1, 1)
        inputs = (a.contiguous(), b.contiguous(), a_scale.contiguous(), b_scale.contiguous())
        if self.bias:
            return (*inputs, torch.randn(self.n, device=run_device(), dtype=self.out_dtype))
        return inputs

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        """The exact int32 product, then the fp32 epilogue and one rounding to ``out_dtype``.

        Every partial sum is an integer below ``127 * 127 * K``, exact in float64.
        """
        a, b, a_scale, b_scale = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        acc = torch.matmul(a.double(), b.double().T).float()
        out = acc * a_scale * b_scale
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


class GEMMW4A16Workload(WorkloadBase):
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
    def from_call(cls, call: Any) -> "GEMMW4A16Workload":
        """The workload of one manifest call of ``GEMMW4A16FwdOp``."""
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
        return torch.matmul(activation.float(), weight.T).to(activation.dtype)

    def verification(self, *inputs):
        from workloads.numerics import Exact

        return Exact(atol=7e-2, rtol=5e-2)


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """The ``[N, K]`` INT4 values of a row-major ``[N, K // 2]`` uint8 packing.

    Byte ``(n, j)`` holds ``k = 2j`` in its low nibble and ``k = 2j + 1`` in its high nibble,
    each a signed two's-complement value in ``[-8, 7]``.
    """
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).to(torch.int16)
    return ((nibbles ^ 8) - 8).reshape(packed.shape[0], 2 * packed.shape[1])


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
    # A 512-byte unit holds 16 rows; a block is 64 rows, the SM90 FP8 WGMMA M extent; a
    # tile spans 128 K.
    unit_rows, block_rows, group_k = 16, 64, 128
    if n % unit_rows or k % group_k:
        raise ValueError(f"N={n} must be a multiple of {unit_rows} and K={k} of {group_k}")
    q = unpack_signed_int4(packed)
    code = torch.where(q > 0, q - 1, torch.where(q == 0, 7, 8 | (-q - 1)))
    parts = []
    for start in range(0, n, block_rows):
        rows = min(block_rows, n - start)
        # Source axes w, hr, r, g, h, p, s, t, i; target g, h, w, r, t, p, hr, s, i. The
        # 32 lanes of a warp are 8 row pairs r by 4 column groups t, and each lane's four
        # FP8 values i fill one 32-bit fragment register.
        block = code[start : start + rows].reshape(
            rows // unit_rows, 2, 8, k // group_k, 2, 2, 2, 4, 4
        )
        nibbles = block.permute(3, 4, 0, 2, 7, 5, 1, 6, 8).reshape(-1, 2)
        parts.append((nibbles[:, 0] | (nibbles[:, 1] << 4)).to(torch.uint8))
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


class GEMMW4A8Workload(WorkloadBase):
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

    # The reference reads row-major weights; a call carries them in the op's repack order.
    reference_takes_call_inputs = False

    def __init__(self, m: int, n: int, k: int, group_size: int) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.group_size = group_size

    @classmethod
    def from_call(cls, call: Any) -> "GEMMW4A8Workload":
        """The workload of one manifest call of ``GEMMW4A8FwdOp``."""
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
        q = torch.randint(-8, 8, (self.n, self.k), device=device, dtype=torch.int16)
        packed = ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.uint8)
        groups = self.k // self.group_size
        group_scale = (0.03 + 0.05 * torch.rand(self.n, groups, device=device)).to(
            torch.float8_e4m3fn
        )
        scale_lower = 0.5
        channel_scale = scale_lower + torch.rand(self.n, device=device)
        token_scale = scale_lower + torch.rand(self.m, device=device)
        return activation, packed, group_scale, channel_scale, token_scale

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


class GEMMW4A8MixedSignWorkload(GEMMW4A8Workload):
    """:class:`GEMMW4A8Workload` with the sign of each group scale drawn at random.

    The signature admits a group scale of either sign; the magnitudes, and so the
    verification, are :class:`GEMMW4A8Workload`'s.
    """

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        activation, packed, group_scale, channel_scale, token_scale = super().gen_inputs()
        negative = torch.rand(group_scale.shape, device=group_scale.device) < 0.5
        signed = torch.where(negative, -group_scale.float(), group_scale.float())
        return activation, packed, signed.to(torch.float8_e4m3fn), channel_scale, token_scale


class BMMWorkload(WorkloadBase):
    """Workload for batched matmul: a=[B,M,K], b=[B,K,N] -> d=[B,M,N]."""

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

    @classmethod
    def from_call(cls, call: Any) -> "BMMWorkload":
        """The workload of one manifest call of ``BMMFwdOp``."""
        indices = call.indices
        return cls(
            indices["B"], indices["M"], indices["N"], indices["K"], getattr(torch, indices["T"])
        )

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device=run_device(), dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a.float(), b.float()).to(a.dtype)

    def torch_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BMMFP8Workload(WorkloadBase):
    """Workload for batched FP8 GEMM.

    ``a`` is ``[B, M, K]``; ``b`` is a contiguous ``[B, K, N]``, or ``[B, N, K]`` under
    ``trans_b``; ``a_scale`` / ``b_scale`` are rank-0 fp32 scalars in ``[0.5, 1.5)``.
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
    def from_call(cls, call: Any) -> "BMMFP8Workload":
        """The workload of one manifest call of ``BMMFP8FwdOp``."""
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
        a = (torch.randn(self.batch, self.m, self.k, device=run_device()) * init_scale).to(
            self.dtype
        )
        b = (torch.randn(self.batch, self.k, self.n, device=run_device()) * init_scale).to(
            self.dtype
        )
        if self.trans_b:
            b = b.transpose(-2, -1).contiguous()
        a_scale = 0.5 + torch.rand((), device=run_device(), dtype=torch.float32)
        b_scale = 0.5 + torch.rand((), device=run_device(), dtype=torch.float32)
        return a, b, a_scale, b_scale

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, a_scale, b_scale = inputs
        if self.trans_b:
            b = b.transpose(-2, -1)
        return torch.bmm(a.float() * a_scale, b.float() * b_scale).to(self.out_dtype)

    def verification(self, *inputs):
        return fp8_matmul_verification(self.k)


class GroupedGEMMWorkload(WorkloadBase):
    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        N: int,
        K: int,
        dtype: torch.dtype,
        trans_a: bool,
        trans_b: bool,
        batch_sizes: list[int] | None = None,
    ):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        if batch_sizes is None:
            # Even groups, the first ``batch_sum % batch_count`` one row longer.
            base, extra = divmod(batch_sum, batch_count)
            batch_sizes = [base + (i < extra) for i in range(batch_count)]
        self.batch_sizes_list = batch_sizes

    @classmethod
    def from_call(cls, call: Any) -> "GroupedGEMMWorkload":
        """The workload of one manifest call of ``GroupedGEMMFwdOp``."""
        indices = call.indices
        return cls(
            indices["M"],
            indices["G"],
            indices["N"],
            indices["K"],
            getattr(torch, indices["T"]),
            indices["trans_a"],
            indices["trans_b"],
            call.metadata_values("batch_sizes"),
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        sizes = self.batch_sizes_list
        N, K = self.N, self.K
        batch_sum, batch_count = sum(sizes), len(sizes)
        if not self.trans_a:
            # NT: B is (batch_count, N, K); NN: (batch_count, K, N).
            a_shape = (batch_sum, K)
            b_shape = (batch_count, N, K) if self.trans_b else (batch_count, K, N)
        else:
            # TT: B is (K, batch_sum); TN: (batch_sum, K).
            a_shape = (batch_sum, N)
            b_shape = (K, batch_sum) if self.trans_b else (batch_sum, K)
        device = run_device()
        A = torch.randn(a_shape, device=device, dtype=self.dtype)
        B = torch.randn(b_shape, device=device, dtype=self.dtype)
        batch_sizes = torch.tensor(sizes, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor([0, *accumulate(sizes[:-1])], device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets

    def ref_program(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        out = self.torch_program(A.float(), B.float(), batch_sizes, batch_offsets)
        return out.to(A.dtype)

    def verification(self, *inputs):
        # NT and NN sum K terms per element; TN and TT sum the rows of one group.
        length = max(self.batch_sizes_list) if self.trans_a else self.K
        return _summation_order_verification(self.dtype, length)

    def torch_program(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        sizes = batch_sizes.tolist()
        offsets = [0, *accumulate(sizes)]
        bounds = list(zip(offsets[:-1], offsets[1:], strict=True))
        if not self.trans_a:
            # NT: A @ B^T; NN: A @ B. The output is (batch_sum, N).
            n = B.shape[1] if self.trans_b else B.shape[2]
            output = torch.empty((sum(sizes), n), device=A.device, dtype=A.dtype)
            for i, (start, end) in enumerate(bounds):
                b = B[i].transpose(0, 1).contiguous() if self.trans_b else B[i]
                output[start:end] = torch.mm(A[start:end], b)
        else:
            # TT: A^T @ B^T; TN: A^T @ B. The output is (batch_count, N, K).
            k = B.shape[0] if self.trans_b else B.shape[1]
            output = torch.zeros((len(sizes), A.shape[1], k), device=A.device, dtype=A.dtype)
            for i, (start, end) in enumerate(bounds):
                b = B[:, start:end].transpose(0, 1) if self.trans_b else B[start:end]
                output[i] = torch.mm(A[start:end].transpose(0, 1), b)
        return output


def fp8_matmul_verification(k):
    """FP8 accumulation noise grows with the square root of the reduction length."""
    from workloads.numerics import Exact, zeroed_input

    return Exact(
        atol=2e-2 * max(1.0, k / 1024) ** 0.5,
        rtol=2e-2,
        controls=(zeroed_input(0, "left-operand-zeroed"),),
    )


class GEMMW4A16BasisWorkload(GEMMW4A16Workload):
    """A basis vector selects exactly one dequantized weight: no reduction error."""

    def __init__(self, k_index):
        super().__init__(1, 35, 384, torch.float16)
        self.k_index = k_index

    def gen_inputs(self):
        n, k, groups = self.n, self.k, self.k // self.group_size
        rows = torch.arange(n)[:, None]
        quantized = torch.randint(0, 16, (n, k), generator=self.rng("packed-weight"))
        zero = ((3 * rows + torch.arange(groups)[None, :]) % 16).to(torch.uint8)
        scale = (
            0.03137 + torch.arange(n * groups, dtype=torch.float32).reshape(n, -1) * 0.001147
        ).to(torch.float16)
        packed = (quantized[:, 0::2] | (quantized[:, 1::2] << 4)).to(torch.uint8)
        group = self.k_index // self.group_size
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
