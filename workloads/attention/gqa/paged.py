import torch

from workloads.attention.gqa.call_metadata import _dtype, _segments
from workloads.attention.gqa.rope import apply_dense_rope
from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device
from workloads.sequence_metadata import make_cu_seqlens
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GQAPagedCall", "GQAPagedFwdWorkload"]


FP8 = torch.float8_e4m3fn


def _cache_scale(scale: torch.Tensor, request: int) -> torch.Tensor:
    """A per-tensor ``[1]`` or per-request, per-KV-head ``[B, H_kv]`` scale of *request*."""
    return scale.view(1, 1, 1) if scale.dim() == 1 else scale[request].view(1, -1, 1)


def _fp8_draw(shape: tuple[int, ...], std: float, device) -> torch.Tensor:
    """Normal values of *std* rounded to ``float8_e4m3fn``."""
    return (torch.randn(shape, device=device) * std).to(FP8)


def _dequant_scale(shape: tuple[int, ...], device) -> torch.Tensor:
    """A float32 dequantization scale drawn from [0.5, 0.8] or [1.25, 2]: at least 20% away
    from one, past every row's tolerance, so a kernel that ignores it disagrees with the
    reference."""
    low = torch.rand(shape, device=device) * 0.3 + 0.5
    return torch.where(torch.rand(shape, device=device) < 0.5, low, 1.0 / low)


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
        has_sinks: bool = False,
        cache_dtype: torch.dtype | None = None,
        per_request_scales: bool = False,
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
        self.has_sinks = has_sinks
        self.cache_dtype = cache_dtype or dtype
        self.per_request_scales = per_request_scales

    @property
    def batch(self) -> int:
        return len(self.q_lens)

    @property
    def table_dtype(self) -> torch.dtype:
        """The output dtype, which the rotary tables share."""
        return self.out_dtype or self.dtype

    def fp8_tensors(
        self, device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, tuple]:
        """Q, K and V pages for the FP8 operands, ``None`` for a 16-bit one, and the scales.

        The score spread in standard deviations is the product of Q's and K's draws: at 2.0
        each, the softmax is far from uniform, so a kernel that drops a scale or the query
        disagrees with the reference. Values draw at 0.5, so with a scale of at most 2 the
        dequantized values stay near one, which the FP8 tolerance assumes.
        """
        pages_shape = (self.num_pages, self.page_size, self.heads_kv, self.dim)
        scale_shape = (self.batch, self.heads_kv) if self.per_request_scales else (1,)
        q = q_scale = k = v = k_scale = v_scale = None
        if self.dtype == FP8:
            q = _fp8_draw((sum(self.q_lens), self.heads, self.dim), 2.0, device)
            q_scale = _dequant_scale((self.batch, self.heads_kv), device)
        if self.cache_dtype == FP8:
            k = _fp8_draw(pages_shape, 2.0, device)
            v = _fp8_draw(pages_shape, 0.5, device)
            k_scale = _dequant_scale(scale_shape, device)
            v_scale = _dequant_scale(scale_shape, device)
        return q, k, v, (q_scale, k_scale, v_scale)

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        """A call over a fragmented page pool, FP8 operands drawn by ``fp8_tensors``."""
        device = run_device()
        pages_shape = (self.num_pages, self.page_size, self.heads_kv, self.dim)
        q8, k8, v8, scales = self.fp8_tensors(device)
        q = (
            q8
            if q8 is not None
            else torch.randn(
                sum(self.q_lens), self.heads, self.dim, dtype=self.dtype, device=device
            )
        )
        k_pages = (
            k8 if k8 is not None else torch.randn(pages_shape, dtype=self.dtype, device=device)
        )
        v_pages = (
            v8 if v8 is not None else torch.randn(pages_shape, dtype=self.dtype, device=device)
        )
        page_table = make_fragmented_block_table(self.batch, self.max_pages_per_req, self.num_pages)
        cache_seqlens = torch.tensor(self.cache_lens, dtype=torch.int32, device=device)
        cos, sin = None, None
        if self.pos_encoding_mode == "rope":
            half = (self.rotary_dim or self.dim) // 2
            frequency = 10000.0 ** (-torch.arange(half, device=device).float() / half)
            angles = torch.arange(max(self.cache_lens), device=device)[:, None] * frequency
            cos, sin = angles.cos().to(self.table_dtype), angles.sin().to(self.table_dtype)
        return (
            q,
            k_pages,
            v_pages,
            page_table,
            cache_seqlens,
            make_cu_seqlens(self.q_lens),
            *scales,
            cos,
            sin,
            torch.randn(self.heads, dtype=torch.float32, device=device) if self.has_sinks else None,
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
        sinks: torch.Tensor | None = None,
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
            if sinks is not None:
                scores = torch.cat((scores, sinks[:, None, None].expand(-1, q_len, 1)), dim=-1)
            probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)[..., :kv_len]
            outputs.append(torch.matmul(probs, v_b).transpose(0, 1))
        return torch.cat(outputs).to(self.table_dtype).contiguous()

    def verification(self, *inputs):
        from workloads.numerics import Exact, reference_tolerance

        if inputs[0].dtype == FP8:
            # Probabilities enter the value product as e4m3, whose 3 mantissa bits round
            # each one by up to 2^-4 of itself; the error is at most 2^-4 of the weighted
            # mean of |v| * v_scale, which these draws keep near one.
            return Exact(rtol=2**-4, atol=2**-4)
        if self.pos_encoding_mode == "rope":
            # Q and K each round their rotated values to the 16-bit table dtype before the
            # score product; the reference retains FP32 rotations, so the budget is twice the
            # dtype's default.
            tol = 2 * reference_tolerance(self.table_dtype)["atol"]
            return Exact(rtol=tol, atol=tol)
        return Exact()


class GQAPagedCall(CallWorkload, GQAPagedFwdWorkload):
    """A manifest call of GQAPagedFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        GQAPagedFwdWorkload.__init__(
            self,
            indices["H"],
            indices["H_kv"],
            indices["D"],
            _segments(call.metadata_values("cu_seqlens_q")),
            call.metadata_values("cache_seqlens"),
            indices["PS"],
            indices["W"],
            indices["NP"],
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
            has_sinks=call.present("sinks"),
            cache_dtype=_dtype(call, "k_pages"),
            per_request_scales=call.present("k_scale") and call.tensors["k_scale"][0] != (1,),
        )

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        """Keep manifest metadata, drawing FP8 operands and their scales by ``fp8_tensors``
        and cosine/sine pairs from rotation angles.

        Independent normal draws for the two tables scale channels as well as rotating
        them, sharpening the softmax instead of representing the declared RoPE workload.
        """
        inputs = list(CallWorkload.gen_inputs(self))
        q8, k8, v8, scales = self.fp8_tensors(inputs[0].device)
        for index, tensor in (
            (0, q8),
            (1, k8),
            (2, v8),
            (6, scales[0]),
            (7, scales[1]),
            (8, scales[2]),
        ):
            if tensor is not None:
                inputs[index] = tensor
        if self.pos_encoding_mode == "rope":
            table = inputs[9]
            angles = torch.rand(
                table.shape, device=table.device, generator=self.rng("rope", device=table.device)
            ) * (2 * torch.pi)
            inputs[9], inputs[10] = angles.cos().to(table.dtype), angles.sin().to(table.dtype)
        return tuple(inputs)
