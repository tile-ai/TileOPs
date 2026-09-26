from typing import Any

import torch

from workloads.workload_base import WorkloadBase


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
        ix = call.ix
        return cls(ix["B"], ix["M"], ix["N"], ix["K"], getattr(torch, ix["T"]))

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device="cuda", dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device="cuda", dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


_FP8_INIT_SCALE: float = 0.25


class BmmFp8Workload(WorkloadBase):
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
    def from_call(cls, call: Any) -> "BmmFp8Workload":
        """The workload of one manifest call of ``BmmFp8FwdOp``."""
        ix = call.ix
        return cls(
            ix["B"],
            ix["M"],
            ix["N"],
            ix["K"],
            getattr(torch, ix["T"]),
            out_dtype=getattr(torch, ix["out_dtype"]),
            trans_b=ix["trans_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (
            (torch.randn(self.batch, self.m, self.k, device="cuda") * _FP8_INIT_SCALE)
            .to(self.dtype)
            .contiguous()
        )
        b = (
            (torch.randn(self.batch, self.k, self.n, device="cuda") * _FP8_INIT_SCALE)
            .to(self.dtype)
            .contiguous()
        )
        if self.trans_b:
            b = b.transpose(-2, -1).contiguous()
        scale_a = (0.5 + torch.rand((), device="cuda", dtype=torch.float32)).contiguous()
        scale_b = (0.5 + torch.rand((), device="cuda", dtype=torch.float32)).contiguous()
        return a, b, scale_a, scale_b

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs
        if self.trans_b:
            b = b.transpose(-2, -1)
        a_f = a.float() * scale_a
        b_f = b.float() * scale_b
        out = torch.bmm(a_f, b_f)
        return out.to(self.out_dtype)
