import torch

from workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128


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


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = W4A16_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine group-wise quantize and pack a logical ``[N, K]`` weight tensor."""
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1)
    group_max = grouped.amax(dim=-1)
    scale = ((group_max - group_min) / 15.0).clamp_min(1e-12)
    zero = torch.round(-group_min / scale).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )

    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = ((quantized.float() - zero.float().unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(
        n, k
    )
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


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

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        activation = torch.randn(self.m, self.k, device="cuda", dtype=self.dtype)
        source_weight = torch.randn(self.n, self.k, device="cuda", dtype=torch.float32) * 0.25
        packed, scale, zero, dequantized = quantize_weight_int4(
            source_weight, group_size=self.group_size
        )
        self._dequantized_weight = dequantized.to(self.dtype).contiguous()
        return activation, packed, scale, zero

    @property
    def dequantized_weight(self) -> torch.Tensor:
        if self._dequantized_weight is None:
            raise RuntimeError("dequantized_weight is available after gen_inputs()")
        return self._dequantized_weight
