import torch

from workloads.attention.gqa.call_metadata import _dtype, _segments
from workloads.attention.gqa.rope import apply_dense_rope
from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device
from workloads.sequence_metadata import make_cu_seqlens
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GQAPagedCall", "GQAPagedFwdWorkload"]


def _cache_scale(scale: torch.Tensor, request: int) -> torch.Tensor:
    """A per-tensor ``[1]`` or per-request, per-KV-head ``[B, H_kv]`` scale of *request*."""
    return scale.view(1, 1, 1) if scale.numel() == 1 else scale[request].view(1, -1, 1)


class GQAPagedFwdWorkload(WorkloadBase):
    """Read-only paged GQA over packed queries and rank-4 KV pages.

    ``cache_lens[b]`` counts request ``b``'s query tokens, which are the last
    ``q_lens[b]`` tokens of its cached sequence: query ``i`` sits at position
    ``cache_lens[b] - q_lens[b] + i``.
    """

    def __init__(
        self,
        heads: int,
        heads_kv: int,
        dim: int,
        q_lens: list[int],
        cache_lens: list[int],
        page_size: int,
        max_pages_per_req: int,
        num_pages: int,
        dtype: torch.dtype,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: float | None = None,
        softcap: float | None = None,
        out_dtype: torch.dtype | None = None,
        pos_encoding_mode: str = "none",
        rotary_dim: int | None = None,
        rope_layout: str = "neox",
    ) -> None:
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.q_lens = q_lens
        self.cache_lens = cache_lens
        self.page_size = page_size
        self.max_pages_per_req = max_pages_per_req
        self.num_pages = num_pages
        self.dtype = dtype
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout

    @property
    def batch(self) -> int:
        return len(self.q_lens)

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        """A 16-bit call over a fragmented page pool."""
        device = run_device()
        pages_shape = (self.num_pages, self.page_size, self.heads_kv, self.dim)
        q = torch.randn(sum(self.q_lens), self.heads, self.dim, dtype=self.dtype, device=device)
        k_pages = torch.randn(pages_shape, dtype=self.dtype, device=device)
        v_pages = torch.randn(pages_shape, dtype=self.dtype, device=device)
        page_table = make_fragmented_block_table(self.batch, self.max_pages_per_req, self.num_pages)
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device=device)
        return (
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            make_cu_seqlens(self.q_lens),
            None,
            None,
            None,
            None,
            None,
        )

    def ref_program(
        self,
        q: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Materialize each request's cached keys, then attend in FP32.

        An FP8 tensor is dequantized by its scale: ``q_scale`` per request and KV head,
        ``k_scale`` / ``v_scale`` per tensor (``[1]``) or per request and KV head. RoPE
        rotates the dequantized query and cached key at their positions in the sequence.
        A query row that sees no key yields zeros.
        """
        groups = self.heads // self.heads_kv
        page_size = k_pages.shape[1]
        scale = self.dim**-0.5 if self.sm_scale is None else self.sm_scale
        rotary_dim = self.dim if self.rotary_dim is None else self.rotary_dim
        bounds = cu_seqlens_q.tolist()
        outputs = []
        for b in range(len(bounds) - 1):
            q_len, kv_len = bounds[b + 1] - bounds[b], int(cache_seqlens[b])
            pages = page_table[b, : -(-kv_len // page_size)].long()
            q_b = q[bounds[b] : bounds[b + 1]].float()
            k_b = k_pages[pages].flatten(0, 1)[:kv_len].float()
            v_b = v_pages[pages].flatten(0, 1)[:kv_len].float()
            if q_scale is not None:
                q_b = q_b * q_scale[b].repeat_interleave(groups).view(1, -1, 1)
            if k_scale is not None:
                k_b = k_b * _cache_scale(k_scale, b)
                v_b = v_b * _cache_scale(v_scale, b)
            q_pos = torch.arange(q_len, device=q.device) + kv_len - q_len
            k_pos = torch.arange(kv_len, device=q.device)
            if self.pos_encoding_mode == "rope":
                rope = dict(rotary_dim=rotary_dim, layout=self.rope_layout)
                q_b = apply_dense_rope(q_b[None], q_pos, rope_cos, rope_sin, **rope)[0]
                k_b = apply_dense_rope(k_b[None], k_pos, rope_cos, rope_sin, **rope)[0]
            k_b = k_b.repeat_interleave(groups, dim=1).transpose(0, 1)
            v_b = v_b.repeat_interleave(groups, dim=1).transpose(0, 1)
            scores = torch.matmul(q_b.transpose(0, 1), k_b.transpose(-2, -1)) * scale
            if self.softcap is not None and self.softcap > 0:
                scores = self.softcap * torch.tanh(scores / self.softcap)
            visible = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device)
            if self.is_causal:
                visible &= k_pos[None, :] <= q_pos[:, None]
            if self.window_size_left >= 0:
                visible &= k_pos[None, :] >= q_pos[:, None] - self.window_size_left
            if self.window_size_right >= 0:
                visible &= k_pos[None, :] <= q_pos[:, None] + self.window_size_right
            scores = scores.masked_fill(~visible, float("-inf"))
            probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
            outputs.append(torch.matmul(probs, v_b).transpose(0, 1))
        return torch.cat(outputs).to(self.out_dtype or q.dtype).contiguous()


class GQAPagedCall(CallWorkload, GQAPagedFwdWorkload):
    """A manifest call of GQAPagedFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        GQAPagedFwdWorkload.__init__(
            self,
            ix["H"],
            ix["H_kv"],
            ix["D"],
            _segments(call.values("cu_seqlens_q")),
            call.values("cache_seqlens"),
            ix["PS"],
            ix["W"],
            ix["NP"],
            _dtype(call, "q"),
            is_causal=params["is_causal"],
            window_size_left=params["window_size_left"],
            window_size_right=params["window_size_right"],
            sm_scale=params["sm_scale"],
            softcap=params["softcap"],
            out_dtype=getattr(torch, params["out_dtype"]) if params["out_dtype"] else None,
            pos_encoding_mode=params["pos_encoding_mode"],
            rotary_dim=params["rotary_dim"],
            rope_layout=params["rope_layout"],
        )

    gen_inputs = CallWorkload.gen_inputs
