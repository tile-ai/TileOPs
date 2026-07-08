import torch

from workloads.workload_base import WorkloadBase


class GemmWorkload(WorkloadBase):

    def __init__(self, m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool = False,
                 trans_b: bool = False):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device='cuda', dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device='cuda', dtype=self.dtype)
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
        a = (torch.randn(self.m, self.k, device='cuda') * 0.25).to(self.dtype).contiguous()
        b = (torch.randn(self.n, self.k, device='cuda') * 0.25).to(self.dtype).contiguous()
        scale_a_shape, scale_b_shape = self._scale_shapes()
        scale_a = (
            0.5 + torch.rand(*scale_a_shape, device='cuda', dtype=torch.float32)
        ).contiguous()
        scale_b = (
            0.5 + torch.rand(*scale_b_shape, device='cuda', dtype=torch.float32)
        ).contiguous()
        if self.bias:
            bias = torch.randn(self.n, device='cuda', dtype=self.out_dtype)
            return a, b, scale_a, scale_b, bias
        return a, b, scale_a, scale_b
