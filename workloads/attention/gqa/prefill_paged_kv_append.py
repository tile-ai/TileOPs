import torch

from workloads.attention.gqa.rope import apply_dense_rope
from workloads.attention.gqa.sequence_metadata import _dtype, _segments, make_cu_seqlens
from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GQAPrefillPagedWithKVCacheFwdWorkload", "GQAPrefillPagedWithKVCacheFwdCall"]


class GQAPrefillPagedWithKVCacheFwdWorkload(WorkloadBase):
    """Packed GQA prefill that appends its keys and values to a paged cache.

    ``cache_lens[b]`` counts request ``b``'s cached tokens before the append: query ``i``
    sits at position ``cache_lens[b] + i``.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        q_lens: list[int],
        cache_lens: list[int],
        page_size: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        fuse_rope: bool = False,
        rotary_dim: int | None = None,
        softcap: float | None = None,
        rope_base: float = 10000.0,
        sm_scale: float | None = None,
    ) -> None:
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
        self.rope_base = rope_base
        self.sm_scale = sm_scale

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
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        q = torch.randn(
            self.total_q, self.heads, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        k_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        v_new = torch.randn(
            self.total_q, self.heads_kv, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        physical_tokens = self.batch * self.max_pages_per_req * self.page_size
        k_pages = torch.randn(
            physical_tokens, self.heads_kv, self.dim, device=run_device(), dtype=self.dtype
        ).contiguous()
        v_pages = torch.randn_like(k_pages)
        cu_seqlens_q = make_cu_seqlens(self.q_lens)
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device=run_device())
        block_table = make_fragmented_block_table(
            self.batch, self.max_pages_per_req, self.batch * self.max_pages_per_req
        )
        return (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )

    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """GPT-NeoX RoPE of packed ``[tokens, heads, dim]`` *x* at *positions*.

        The table is rounded to x's dtype, as the op's is, and the rotation runs in FP32.
        """
        rotary_dim = self.dim if self.rotary_dim is None else self.rotary_dim
        half = rotary_dim // 2
        inv_freq = 1.0 / (
            self.rope_base ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
        )
        angles = torch.outer(positions.float(), inv_freq)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        index = torch.arange(positions.numel(), device=x.device)
        return apply_dense_rope(x[None], index, cos, sin, rotary_dim=rotary_dim, layout="neox")[0]

    def ref_program(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Attend each request's chunk to its cached keys plus its own, in FP32, then append
        the chunk's keys and values to the pages after the cached ones.

        An FP8 cache is dequantized by its scale to q's dtype, and the appended rows are
        quantized by it; an unquantized cache ignores the scales. Fused RoPE rotates the
        new queries and keys at their positions; the cached keys are stored rotated.
        A query row that sees no key yields zeros.
        """
        groups = self.heads // self.heads_kv
        page_size = self.page_size
        scale = self.dim**-0.5 if self.sm_scale is None else self.sm_scale
        quantized = k_pages.dtype == torch.float8_e4m3fn
        bounds = cu_seqlens_q.tolist()
        outputs, appends = [], []
        for b in range(len(bounds) - 1):
            start, end = bounds[b], bounds[b + 1]
            q_len, old_len = end - start, int(cache_seqlens[b])
            old_pos = torch.arange(old_len, device=q.device)
            rows = block_table[b, old_pos // page_size].long() * page_size + old_pos % page_size
            k_old, v_old = k_pages[rows], v_pages[rows]
            if quantized:
                k_old = (k_old.float() * k_scale[0]).to(q.dtype)
                v_old = (v_old.float() * v_scale[0]).to(q.dtype)
            q_b, k_b = q[start:end], k_new[start:end]
            q_pos = torch.arange(q_len, device=q.device) + old_len
            if self.fuse_rope:
                q_b, k_b = self.rope(q_b, q_pos), self.rope(k_b, q_pos)
            new_rows = block_table[b, q_pos // page_size].long() * page_size + q_pos % page_size
            appends.append((new_rows, k_b, v_new[start:end]))
            k_all = torch.cat([k_old, k_b]).repeat_interleave(groups, dim=1).transpose(0, 1)
            v_all = torch.cat([v_old, v_new[start:end]]).repeat_interleave(groups, dim=1)
            scores = torch.matmul(q_b.transpose(0, 1).float(), k_all.float().transpose(-2, -1))
            scores = scores * scale
            if self.softcap is not None and self.softcap > 0:
                scores = self.softcap * torch.tanh(scores / self.softcap)
            if self.is_causal:
                kv_pos = torch.arange(old_len + q_len, device=q.device)
                visible = kv_pos[None, :] <= q_pos[:, None]
                scores = scores.masked_fill(~visible, float("-inf"))
            probs = torch.softmax(scores, dim=-1).nan_to_num()
            out = torch.matmul(probs, v_all.transpose(0, 1).float()).transpose(0, 1)
            outputs.append(out.to(q.dtype))
        for rows, k_b, v_b in appends:
            if quantized:
                k_b = (k_b.float() / k_scale[0]).to(k_pages.dtype)
                v_b = (v_b.float() / v_scale[0]).to(v_pages.dtype)
            k_pages[rows], v_pages[rows] = k_b, v_b
        return torch.cat(outputs).contiguous()


class GQAPrefillPagedWithKVCacheFwdCall(CallWorkload, GQAPrefillPagedWithKVCacheFwdWorkload):
    """A manifest call of GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp.

    An FP8 cache holds the random pages quantized by scales of 0.01; an unquantized one
    passes unit scales.
    """

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        q_lens = _segments(call.values("cu_seqlens_q"))
        GQAPrefillPagedWithKVCacheFwdWorkload.__init__(
            self,
            len(q_lens),
            ix["H"],
            ix["H_kv"],
            q_lens,
            call.values("cache_seqlens"),
            params["page_size"],
            ix["D"],
            params["is_causal"],
            _dtype(call, "q"),
            fuse_rope=params["fuse_rope"],
            rotary_dim=params["rotary_dim"],
            softcap=params["softcap"],
            rope_base=params["rope_base"],
            sm_scale=params["sm_scale"],
        )
        self.cache_dtype = _dtype(call, "k_pages")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        q, k_new, v_new, k_pages, v_pages, cu_seqlens_q, cache_seqlens, block_table = (
            GQAPrefillPagedWithKVCacheFwdWorkload.gen_inputs(self)
        )
        scale = 1.0
        if self.cache_dtype == torch.float8_e4m3fn:
            scale = 0.01
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            k_pages, v_pages = (
                (t / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn).contiguous()
                for t in (k_pages, v_pages)
            )
        k_scale = torch.full((1,), scale, dtype=torch.float32, device=q.device)
        return (
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            k_scale.clone(),
            cu_seqlens_q,
            cache_seqlens,
            block_table,
        )
