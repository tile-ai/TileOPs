import torch

from workloads.workload_base import WorkloadBase


class GQAPrefillFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len_q: int,
                 seq_len_kv: int, dim: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len_q, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v = torch.randn(
            self.batch, self.seq_len_kv, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        return q, k, v


class GQAPrefillVarlenFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, q_lens: list[int],
                 kv_lens: list[int], dim: int, is_causal: bool,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.q_lens = q_lens
        self.kv_lens = kv_lens
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def total_kv(self) -> int:
        return sum(self.kv_lens)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.q_lens)

    @property
    def max_seqlen_kv(self) -> int:
        return max(self.kv_lens)

    def gen_inputs(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.total_q, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v = torch.randn(
            self.total_kv, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device='cuda')
        cu_seqlens_kv = torch.tensor(
            [0] + torch.tensor(self.kv_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device='cuda')
        return q, k, v, cu_seqlens_q, cu_seqlens_kv


class GQAPrefillWithKVCacheFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len_new: int,
                 seq_len_cap: int, dim: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_new = seq_len_new
        self.seq_len_cap = seq_len_cap
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len_new, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k_new = torch.randn(
            self.batch, self.seq_len_new, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v_new = torch.randn(
            self.batch, self.seq_len_new, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k_cache = torch.randn(
            self.batch, self.seq_len_cap, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v_cache = torch.randn(
            self.batch, self.seq_len_cap, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        old_len = self.seq_len_cap - self.seq_len_new
        cache_seqlens = torch.full(
            (self.batch,), old_len, dtype=torch.int32, device='cuda')
        return q, k_new, v_new, k_cache, v_cache, cache_seqlens
