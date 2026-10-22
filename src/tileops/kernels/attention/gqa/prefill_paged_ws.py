"""SM90 paged GQA prefill in two launches: append the chunk, then attend over the grown cache.

The append launch writes each request's new keys and values into the pages past its cache
length: the keys rotated under fused RoPE, both quantized by the pool's scale on an FP8 pool.
It rotates the queries under fused RoPE and writes each request's length after the append.
The attention is the paged warp-specialized program of ``GQAPagedFwdOp`` over that cache,
whose last positions are the chunk's queries.
"""

import dataclasses
import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.call_spec import (
    ATTENTION_DTYPES,
    AttentionCall,
    GQAPrefillPagedFwdInterface,
)
from tileops.kernels.attention.gqa.paged_fp8_ws import GQAPagedFP8WSKernel
from tileops.kernels.attention.gqa.paged_ws import GQAPagedFwdWSKernel
from tileops.kernels.constants import FP8_E4M3_MAX
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GQAPrefillPagedWSKernel"]


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def _gqa_prefill_paged_append_kernel(
    batch, heads, heads_kv, dim, page_size, max_pages_per_req, rotary_dim, dtype, cache_dtype
):
    """Append the chunk to the pages; with ``rotary_dim`` rotate it and the queries first.

    One block writes 8 new tokens of one request and KV head, and rotates the same tokens of
    that head's query group. ``rotary_dim`` 0 rotates nothing and leaves ``Qout`` unwritten.
    """
    block_m = 8
    group = heads // heads_kv
    half = rotary_dim // 2
    fp8 = cache_dtype == "float8_e4m3fn"
    total_q = T.dynamic("total_q")
    pool_rows = T.dynamic("pool_rows")
    positions = T.dynamic("positions")

    def store(value, scale):
        """A value in the pool's element type: quantized by the pool's scale on an FP8 pool."""
        if fp8:
            return T.Cast(cache_dtype, T.clamp(value / scale, -FP8_E4M3_MAX, FP8_E4M3_MAX))
        return T.Cast(cache_dtype, value)

    @T.prim_func
    def main(
        Q: T.Tensor((total_q, heads, dim), dtype),  # type: ignore
        Knew: T.Tensor((total_q, heads_kv, dim), dtype),  # type: ignore
        Vnew: T.Tensor((total_q, heads_kv, dim), dtype),  # type: ignore
        Kpool: T.Tensor((pool_rows, heads_kv, dim), cache_dtype),  # type: ignore
        Vpool: T.Tensor((pool_rows, heads_kv, dim), cache_dtype),  # type: ignore
        Kscale: T.Tensor((1,), "float32"),  # type: ignore
        Vscale: T.Tensor((1,), "float32"),  # type: ignore
        CuQ: T.Tensor((batch + 1,), "int32"),  # type: ignore
        Lens: T.Tensor((batch,), "int32"),  # type: ignore
        Table: T.Tensor((batch, max_pages_per_req), "int32"),  # type: ignore
        Cos: T.Tensor((positions, max(half, 1)), dtype),  # type: ignore
        Sin: T.Tensor((positions, max(half, 1)), dtype),  # type: ignore
        Qout: T.Tensor((total_q, heads, dim), dtype),  # type: ignore
        Total: T.Tensor((batch,), "int32"),  # type: ignore
        max_seqlen_q: T.int32,  # type: ignore
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, block_m), heads_kv, batch, threads=128) as (
            bx,
            hk,
            b,
        ):
            q_start = CuQ[b]
            q_len = CuQ[b + 1] - q_start
            old_len = Lens[b]
            k_scale = Kscale[0] if fp8 else 1.0
            v_scale = Vscale[0] if fp8 else 1.0
            if (bx == 0) & (hk == 0) & (T.get_thread_binding() == 0):
                Total[b] = old_len + q_len
            # Neox RoPE pairs column d with d + half below rotary_dim: each pair is computed
            # once, from two contiguous runs of the row, and the columns past it are copied.
            if rotary_dim > 0:
                for i, d in T.Parallel(block_m, half):
                    p = bx * block_m + i
                    if p < q_len:
                        pos = old_len + p
                        row = Table[b, pos // page_size] * page_size + pos % page_size
                        c = T.Cast("float32", Cos[pos, d])
                        s = T.Cast("float32", Sin[pos, d])
                        x = T.Cast("float32", Knew[q_start + p, hk, d])
                        y = T.Cast("float32", Knew[q_start + p, hk, d + half])
                        Kpool[row, hk, d] = store(x * c - y * s, k_scale)
                        Kpool[row, hk, d + half] = store(y * c + x * s, k_scale)
            if rotary_dim < dim:
                for i, d in T.Parallel(block_m, dim - rotary_dim):
                    p = bx * block_m + i
                    if p < q_len:
                        pos = old_len + p
                        row = Table[b, pos // page_size] * page_size + pos % page_size
                        key = T.Cast("float32", Knew[q_start + p, hk, rotary_dim + d])
                        Kpool[row, hk, rotary_dim + d] = store(key, k_scale)
            for i, d in T.Parallel(block_m, dim):
                p = bx * block_m + i
                if p < q_len:
                    pos = old_len + p
                    row = Table[b, pos // page_size] * page_size + pos % page_size
                    Vpool[row, hk, d] = store(T.Cast("float32", Vnew[q_start + p, hk, d]), v_scale)
            if rotary_dim > 0:
                for i, h, d in T.Parallel(block_m, group, half):
                    p = bx * block_m + i
                    if p < q_len:
                        pos = old_len + p
                        head = hk * group + h
                        c = T.Cast("float32", Cos[pos, d])
                        s = T.Cast("float32", Sin[pos, d])
                        x = T.Cast("float32", Q[q_start + p, head, d])
                        y = T.Cast("float32", Q[q_start + p, head, d + half])
                        Qout[q_start + p, head, d] = T.Cast(dtype, x * c - y * s)
                        Qout[q_start + p, head, d + half] = T.Cast(dtype, y * c + x * s)
            if 0 < rotary_dim < dim:
                for i, h, d in T.Parallel(block_m, group, dim - rotary_dim):
                    p = bx * block_m + i
                    if p < q_len:
                        head = hk * group + h
                        Qout[q_start + p, head, rotary_dim + d] = Q[
                            q_start + p, head, rotary_dim + d
                        ]

    return main


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    compile_flags=["-O3", "-DENABLE_BF16"],
)
def _gqa_prefill_paged_image_kernel(batch, heads_kv, dim, page_size, max_pages_per_req, dtype):
    """Copy each request's pages, up to its length after the append, out of an FP8 pool into
    a 16-bit image: page w of request b lands on image page b * max_pages_per_req + w,
    each value multiplied by the pool's scale.

    A page is one contiguous run of the pool, so each thread converts 16 bytes a step of
    the flat views; the key and the value page of a request's page are separate blocks.
    The head dimension is a multiple of 16, so a step never splits a row's 16 bytes.
    """
    threads = 256
    vec = 16
    page_elems = page_size * heads_kv * dim
    pool_elems = T.dynamic("pool_elems")
    image_elems = batch * max_pages_per_req * page_elems

    @T.prim_func
    def main(
        Kpool: T.Tensor((pool_elems,), "float8_e4m3fn"),  # type: ignore
        Vpool: T.Tensor((pool_elems,), "float8_e4m3fn"),  # type: ignore
        Kscale: T.Tensor((1,), "float32"),  # type: ignore
        Vscale: T.Tensor((1,), "float32"),  # type: ignore
        Total: T.Tensor((batch,), "int32"),  # type: ignore
        Table: T.Tensor((batch, max_pages_per_req), "int32"),  # type: ignore
        Kimage: T.Tensor((image_elems,), dtype),  # type: ignore
        Vimage: T.Tensor((image_elems,), dtype),  # type: ignore
    ):
        with T.Kernel(max_pages_per_req, batch, 2, threads=threads) as (w, b, kv):
            tx = T.get_thread_binding()
            if w * page_size < Total[b]:
                src = T.alloc_var("int64", init=T.Cast("int64", Table[b, w]) * page_elems)
                dst = T.Cast("int64", b * max_pages_per_req + w) * page_elems
                if kv == 0:
                    scale = T.alloc_var("float32", init=Kscale[0])
                    for step in T.serial(T.ceildiv(page_elems, threads * vec)):
                        e = (step * threads + tx) * vec
                        if e < page_elems:
                            for v in T.vectorized(vec):
                                Kimage[dst + e + v] = T.Cast(
                                    dtype, T.Cast("float32", Kpool[src + e + v]) * scale
                                )
                else:
                    scale = T.alloc_var("float32", init=Vscale[0])
                    for step in T.serial(T.ceildiv(page_elems, threads * vec)):
                        e = (step * threads + tx) * vec
                        if e < page_elems:
                            for v in T.vectorized(vec):
                                Vimage[dst + e + v] = T.Cast(
                                    dtype, T.Cast("float32", Vpool[src + e + v]) * scale
                                )

    return main


class GQAPrefillPagedWSKernel(Kernel, GQAPrefillPagedFwdInterface):
    """SM90 paged prefill: an append launch, then the paged warp-specialized attention.

    It serves the calls the paged warp-specialized program serves over the grown cache: a
    16-bit cache, with or without fused RoPE, or an FP8 cache without it.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset(
        {
            "gqa_prefill_paged_with_kv_cache_fwd_kernel",
            "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
        }
    )

    # The temporary memory an FP8 call may take for the 16-bit image of its pages; a call
    # that needs more is left to the kernels that read the pool in place.
    _MAX_IMAGE_BYTES: int = 4 << 30

    @classmethod
    def _image_bytes(cls, call: AttentionCall) -> int:
        """The key and value image of every page the block table can name."""
        rows = call.batch * call.max_pages_per_req * call.page_size
        return 2 * rows * call.heads_kv * call.dim * call.dtype.itemsize

    @staticmethod
    def _attention(call: AttentionCall) -> "tuple[type[Kernel], AttentionCall, bool]":
        """The paged program, the call it serves once the chunk is in the cache, rotated, and
        whether an FP8 pool is read through a 16-bit image.

        The FP8 program converts tiles in the kernel where it serves the head dimension;
        elsewhere the image costs one pass over the pages and the 16-bit program reads it.
        """
        attend = dataclasses.replace(call, fuse_rope=False, max_position=None, rotary_dim=None)
        if call.cache_dtype == torch.float8_e4m3fn:
            if GQAPagedFP8WSKernel.refusal(attend) is None:
                return GQAPagedFP8WSKernel, attend, False
            return GQAPagedFwdWSKernel, dataclasses.replace(attend, cache_dtype=call.dtype), True
        return GQAPagedFwdWSKernel, attend, False

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.dtype not in ATTENTION_DTYPES:
            return "requires a float16 or bfloat16 query"
        if call.cache_dtype not in (call.dtype, torch.float8_e4m3fn):
            return "requires a cache of the query's dtype or FP8"
        if call.fuse_rope and call.cache_dtype != call.dtype:
            return "does not serve fused RoPE over an FP8 cache"
        attend, attend_call, image = cls._attention(call)
        reason = attend.refusal(attend_call)
        if reason is not None:
            return f"its attention {reason}"
        if image and cls._image_bytes(call) > cls._MAX_IMAGE_BYTES:
            return f"the 16-bit image of the FP8 pool exceeds {cls._MAX_IMAGE_BYTES} bytes"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: AttentionCall, config: Optional[dict] = None) -> None:
        """Build the append program and the attention for the call's geometry.

        Args:
            call: The paged prefill call this object serves.
            config: Unused; both launches fix their own tiles.
        """
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.dtype = call.dtype
        self.attention, attend_call, self.reads_image = self._attention(call)
        if self.attention is GQAPagedFP8WSKernel:
            self._program = GQAPagedFP8WSKernel.program(
                call.batch, call.heads, call.heads_kv, call.dim, call.page_size,
                call.max_pages_per_req, call.is_causal, call.sm_scale, call.softcap,
                self.dtype_str, self.dtype_str, False,
                GQAPagedFP8WSKernel.tiling_for(attend_call), self.device_index,
            )  # fmt: skip
            self._split_chunks = GQAPagedFP8WSKernel.split_chunks_for(
                call.batch,
                call.heads_kv,
                call.dim,
                call.max_pages_per_req * call.page_size,
                GQAPagedFP8WSKernel.tiling_for(attend_call)[0],
                False,
            )
        else:
            self._program = GQAPagedFwdWSKernel.program(
                call.batch, call.heads, call.heads_kv, call.dim, call.page_size,
                call.max_pages_per_req, call.is_causal, call.sm_scale, call.softcap,
                self.dtype_str, self.device_index,
            )  # fmt: skip
        self._counters: dict = {}
        self._rotary_dim = call.rotary_dim if call.fuse_rope else 0
        self._append = _gqa_prefill_paged_append_kernel(
            call.batch,
            call.heads,
            call.heads_kv,
            call.dim,
            call.page_size,
            call.max_pages_per_req,
            self._rotary_dim,
            self.dtype_str,
            self.dtype_to_str(call.cache_dtype),
        )
        device = torch.device("cuda", self.device_index)
        self._fp8 = call.cache_dtype == torch.float8_e4m3fn
        if self.reads_image:
            self._image = _gqa_prefill_paged_image_kernel(
                call.batch, call.heads_kv, call.dim, call.page_size, call.max_pages_per_req,
                self.dtype_str,
            )  # fmt: skip
            pages = call.batch * call.max_pages_per_req
            self._image_table = torch.arange(pages, dtype=torch.int32, device=device).view(
                call.batch, call.max_pages_per_req
            )
            self._image_shape = (pages * call.page_size, call.heads_kv, call.dim)
        # Stand-ins for the tables, scales and sinks a call without them never reads.
        self._no_table = torch.zeros((1, 1), dtype=self.dtype, device=device)
        self._unit_scale = torch.ones(1, dtype=torch.float32, device=device)
        self._unit_q_scale = torch.ones(
            (call.batch, call.heads_kv), dtype=torch.float32, device=device
        )
        self._no_sinks = torch.empty(call.heads, dtype=torch.float32, device=device)
        self.init_config(config)

    def forward(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        max_seqlen_q: int,
        cos_table: Optional[torch.Tensor] = None,
        sin_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_scale = k_scale if self._fp8 else self._unit_scale
        v_scale = v_scale if self._fp8 else self._unit_scale
        q_attend = torch.empty_like(q) if self._rotary_dim else q
        total = torch.empty_like(cache_seqlens)
        self._append(
            q,
            k_new,
            v_new,
            k_pages,
            v_pages,
            k_scale,
            v_scale,
            cu_seqlens_q,
            cache_seqlens,
            block_table,
            cos_table if self._rotary_dim else self._no_table,
            sin_table if self._rotary_dim else self._no_table,
            q_attend,
            total,
            max_seqlen_q,
        )
        if self.attention is GQAPagedFP8WSKernel:
            return GQAPagedFP8WSKernel.launch(
                self._program, self._counters, q_attend, k_pages, v_pages, total, block_table,
                cu_seqlens_q, self._unit_q_scale, k_scale, v_scale, self.dtype, self._split_chunks,
            )  # fmt: skip
        k_read, v_read, table_read = k_pages, v_pages, block_table
        if self.reads_image:
            k_read = torch.empty(self._image_shape, dtype=self.dtype, device=q.device)
            v_read = torch.empty_like(k_read)
            self._image(
                k_pages.view(-1), v_pages.view(-1), k_scale, v_scale, total, block_table,
                k_read.view(-1), v_read.view(-1),
            )  # fmt: skip
            table_read = self._image_table
        return GQAPagedFwdWSKernel.launch(
            self._program, self._counters, q_attend, k_read, v_read, total, table_read,
            cu_seqlens_q, self._no_sinks,
        )  # fmt: skip
