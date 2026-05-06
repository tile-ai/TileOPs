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
                 seq_len_cap: int, dim: int, is_causal: bool, dtype: torch.dtype,
                 fuse_rope: bool = False, rotary_dim: int | None = None,
                 softcap: float | None = None) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_new = seq_len_new
        self.seq_len_cap = seq_len_cap
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.fuse_rope = fuse_rope
        self.rotary_dim = rotary_dim
        self.softcap = softcap

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


class GQAPrefillPagedWithKVCacheFwdTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, heads_kv: int, q_lens: list[int],
                 cache_lens: list[int], page_size: int, dim: int, is_causal: bool,
                 dtype: torch.dtype, fuse_rope: bool = False,
                 rotary_dim: int | None = None, softcap: float | None = None) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.q_lens = q_lens
        self.cache_lens = cache_lens
        self.page_size = page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.fuse_rope = fuse_rope
        self.rotary_dim = rotary_dim
        self.softcap = softcap

    @property
    def total_q(self) -> int:
        return sum(self.q_lens)

    @property
    def max_seqlen_q(self) -> int:
        return max(self.q_lens)

    @property
    def max_total_len(self) -> int:
        return max(cache + q for cache, q in zip(self.cache_lens, self.q_lens, strict=True))

    @property
    def max_pages_per_req(self) -> int:
        return (self.max_total_len + self.page_size - 1) // self.page_size

    def gen_inputs(
        self
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, int]:
        q = torch.randn(
            self.total_q, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        k_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        physical_tokens = self.batch * self.max_pages_per_req * self.page_size
        k_pages = torch.randn(
            physical_tokens, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        v_pages = torch.randn_like(k_pages)
        cu_seqlens_q = torch.tensor(
            [0] + torch.tensor(self.q_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device='cuda')
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device='cuda')
        block_table = torch.arange(
            self.batch * self.max_pages_per_req, dtype=torch.int32,
            device='cuda').reshape(self.batch, self.max_pages_per_req).contiguous()
        return (
            q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens,
            block_table, self.max_seqlen_q)
