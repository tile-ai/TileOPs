import torch

from workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128
# Lanes sharing a weight row in the A fragment the prepacked GEMM reads. Fixes
# the permutation below, so it is not a knob: another value is another layout.
W4A16_REPACK_LANES = 4


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

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device="cuda", dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device="cuda", dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFp8Workload(WorkloadBase):
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

    def _scale_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.scale_mode in ("per_tensor", "tensor"):
            return (1, 1), (1, 1)
        if self.scale_mode == "block128":
            if self.k % 128 != 0:
                raise ValueError("block128 FP8 workloads require k divisible by 128")
            return (self.m, self.k // 128), (self.n, self.k // 128)
        raise ValueError(f"unknown FP8 GEMM scale_mode {self.scale_mode!r}")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (torch.randn(self.m, self.k, device="cuda") * 0.25).to(self.dtype).contiguous()
        b = (torch.randn(self.n, self.k, device="cuda") * 0.25).to(self.dtype).contiguous()
        scale_a_shape, scale_b_shape = self._scale_shapes()
        scale_a = (
            0.5 + torch.rand(*scale_a_shape, device="cuda", dtype=torch.float32)
        ).contiguous()
        scale_b = (
            0.5 + torch.rand(*scale_b_shape, device="cuda", dtype=torch.float32)
        ).contiguous()
        if self.bias:
            bias = torch.randn(self.n, device="cuda", dtype=self.out_dtype)
            return a, b, scale_a, scale_b, bias
        return a, b, scale_a, scale_b

    def _expand_scale(self, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        if tuple(scale.shape) == (1, 1):
            return scale.expand(rows, cols)
        scale_cols = (cols + 127) // 128
        if tuple(scale.shape) != (rows, scale_cols):
            raise ValueError(f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")
        return scale.repeat_interleave(128, dim=1)[:, :cols]

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(scale_a, self.m, self.k)
        b_f = b.float() * self._expand_scale(scale_b, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)


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
    # The scale is stored in `scale_dtype`, so round to it before deriving
    # anything from it: the zero point, the nibbles and the dequantized
    # reference all have to describe the weight the kernel reconstructs from
    # the stored value. Clamping after the cast keeps a degenerate all-zero
    # group on the smallest normal of that dtype rather than at an FP32
    # epsilon that flushes to zero in FP16.
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


def repack_w4a16_weight(packed: torch.Tensor, step_k: int = 128) -> torch.Tensor:
    """Reorder each K step of a ``[N, K/2]`` packed weight for the A fragment.

    The shape is unchanged. Within every ``step_k // 2`` bytes the nibbles are
    permuted so that a lane reads one contiguous 32-bit word and
    ``tileops_w4a16_dequant_word`` turns it into eight weights with four LOP3s:
    word ``(c, wd)`` takes nibble ``j`` from the low half of source byte
    ``16*wd + 4*j + c`` and nibble ``j+4`` from its high half, which is exactly
    the pair fragment positions ``32*wd + 8*j + 2*c + {0,1}`` want.

    Args:
        packed: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``.
        step_k: Weights per MMA step; ``step_k // 2`` must be a multiple of 16.

    Returns:
        A contiguous ``[N, K/2]`` ``torch.uint8`` tensor in the layout
        :class:`GemmW4A16Kernel` reads.

    Example:
        >>> prepacked = repack_w4a16_weight(packed_weight)
    """
    n, kp = packed.shape
    step = step_k // 2
    if kp % step or step % (4 * W4A16_REPACK_LANES):
        raise ValueError(
            f"K/2={kp} must be a multiple of step_k/2={step}, which must itself be"
            f" a multiple of {4 * W4A16_REPACK_LANES}"
        )
    n_steps = kp // step
    words_per_lane = step // (4 * W4A16_REPACK_LANES)
    grouped = packed.view(n, n_steps, step).to(torch.int32)
    low, high = grouped & 0xF, (grouped >> 4) & 0xF
    out = torch.zeros(n, n_steps, step // 4, dtype=torch.int32, device=packed.device)
    for lane in range(W4A16_REPACK_LANES):
        for word in range(words_per_lane):
            acc = torch.zeros(n, n_steps, dtype=torch.int32, device=packed.device)
            for pair in range(4):
                src = 16 * word + 4 * pair + lane
                acc = acc | (low[:, :, src] << (4 * pair)) | (high[:, :, src] << (4 * pair + 16))
            out[:, :, lane * words_per_lane + word] = acc
    return out.reshape(n, kp // 4).view(torch.uint8).reshape(n, kp).contiguous()


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
        self._dequantized_weight: torch.Tensor | None = None
        self._row_major_weight: torch.Tensor | None = None

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The op's inputs, with the weight already in the order it reads.

        The repack is what a serving stack does once when a checkpoint loads, so
        it belongs outside anything timed -- which is also what makes a
        comparison against Marlin or Machete even: each of them builds its own
        layout here too.
        """
        activation = torch.randn(self.m, self.k, device="cuda", dtype=self.dtype)
        source_weight = torch.randn(self.n, self.k, device="cuda", dtype=torch.float32) * 0.25
        packed, scale, zero, dequantized = quantize_weight_int4(
            source_weight, group_size=self.group_size, scale_dtype=self.dtype
        )
        self._dequantized_weight = dequantized.to(self.dtype).contiguous()
        self._row_major_weight = packed
        return activation, repack_w4a16_weight(packed), scale, zero

    @property
    def dequantized_weight(self) -> torch.Tensor:
        if self._dequantized_weight is None:
            raise RuntimeError("dequantized_weight is available after gen_inputs()")
        return self._dequantized_weight

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
        del packed_weight, weight_scale, weight_zero
        return torch.matmul(activation, self.dequantized_weight.T)
