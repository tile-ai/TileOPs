import torch

from workloads.workload_base import WorkloadBase


class GatedDeltaNetFwdTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        q = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, S, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, S, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, S, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, S, device="cuda", dtype=self.dtype) * 0.5
        return q, k, v, g, beta


class GatedDeltaNetPrefillFwdTest(GatedDeltaNetFwdTest):
    """Inference prefill workload for Gated DeltaNet."""

    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int,
        dtype: torch.dtype,
        layout: str = "bhtd",
    ) -> None:
        super().__init__(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
        self.layout = self._normalize_layout(layout)
        if self.layout == "bthd":
            self.shape = (batch, seq_len, heads, dim_k)
        else:
            self.shape = (batch, heads, seq_len, dim_k)

    @staticmethod
    def _normalize_layout(layout: str) -> str:
        layout = layout.lower()
        if layout == "bhsd":
            return "bhtd"
        if layout in ("bhtd", "bthd"):
            return layout
        raise ValueError(f"Unsupported layout: {layout}")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        if self.layout != "bthd":
            return super().gen_inputs()

        B, H, S, DK, DV = self.batch, self.heads, self.seq_len, self.dim_k, self.dim_v
        q = torch.randn(B, S, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, S, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, S, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, S, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, S, H, device="cuda", dtype=self.dtype) * 0.5
        return q, k, v, g, beta


class GatedDeltaNetDecodeTest(WorkloadBase):

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device="cuda", dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device="cuda", dtype=self.dtype) * 0.1
        g = -torch.rand(B, H, device="cuda", dtype=self.dtype)
        beta = torch.rand(B, H, device="cuda", dtype=self.dtype) * 0.5
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=self.dtype) * 0.1
        return q, k, v, g, beta, state
