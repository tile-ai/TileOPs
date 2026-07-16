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

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device="cuda", dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device="cuda", dtype=self.dtype)
        return a, b

_FP8_INIT_SCALE: float = 0.25


class BmmFp8Workload(WorkloadBase):
    """Workload for batched FP8 GEMM.

    Layout contract
    ---------------
    * ``a`` is produced in ``[B, M, K]`` (K-contiguous, matches manifest input
      ``a`` signature).
    * ``b`` is produced in ``[B, K, N]`` (N-contiguous).  This is the *KN*
      layout advertised by the manifest as the primary path.  The alternate
      ``[B, N, K]`` layout accepted by ``BmmFp8Op`` is exercised directly in
      ``tests/ops/test_bmm.py`` (see ``test_bmm_fp8_accepts_nk_layout_when_k_ne_n``),
      not through this workload.
    * ``scale_a`` / ``scale_b`` are per-tensor rank-0 fp32 scalars in
      ``[0.5, 1.5)``.
    """

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (
            torch.randn(self.batch, self.m, self.k, device="cuda") * _FP8_INIT_SCALE
        ).to(self.dtype).contiguous()
        b = (
            torch.randn(self.batch, self.k, self.n, device="cuda") * _FP8_INIT_SCALE
        ).to(self.dtype).contiguous()
        # per_tensor uses 0-D scalars (empty shape); torch.rand accepts an
        # empty size tuple and produces a rank-0 tensor.
        scale_a = (
            0.5 + torch.rand((), device="cuda", dtype=torch.float32)
        ).contiguous()
        scale_b = (
            0.5 + torch.rand((), device="cuda", dtype=torch.float32)
        ).contiguous()
        return a, b, scale_a, scale_b
