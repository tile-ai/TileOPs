"""Hopper paged attention with request-local prefill and packed-head decode paths."""

import ctypes
import functools
import sys

import torch

from tileops.kernels.attention.call_spec import GQAPagedFwdInterface
from tileops.kernels.attention.gqa.paged import GQAPagedFwdKernel
from tileops.kernels.attention.gqa.paged_unified import paged_unified_kernel
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_count

__all__ = ["GQAPagedMixedKernel"]


@functools.lru_cache(maxsize=1)
def _memset_d32_async():
    """Bind the driver memset without adding a CUDA Python runtime dependency."""
    driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
    memset = driver.cuMemsetD32Async
    memset.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_size_t, ctypes.c_void_p]
    memset.restype = ctypes.c_int
    return memset


def _new_counters(count, device):
    """Use a memset graph node instead of an additional PyTorch fill kernel."""
    counter = torch.empty(count, dtype=torch.int32, device=device)
    with torch.cuda.device(device):
        status = _memset_d32_async()(
            counter.data_ptr(), 0, count, torch.cuda.current_stream(device).cuda_stream
        )
    if status:
        raise RuntimeError(f"cuMemsetD32Async failed with CUDA driver error {status}")
    return counter


class GQAPagedMixedKernel(Kernel, GQAPagedFwdInterface):
    """Run live ragged requests and split reduction in one CUDA kernel.

    Small packed totals use one query per CTA. Larger calls share one persistent
    queue, packing query heads for short requests. Both geometries read live
    offsets on every replay and retain split reduction inside the same launch.
    """

    supported_archs = [90]
    preferred_over = frozenset({"gqa_paged_varlen_kernel"})

    @classmethod
    def refusal(cls, call):
        reason = GQAPagedFwdKernel.refusal(call)
        if reason is not None:
            return reason
        if call.fuse_rope:
            return "uses the general paged implementation for RoPE"
        if call.sm_scale is not None and call.sm_scale <= 0:
            return "requires a positive softmax scale"
        if call.dim not in (64, 128) or call.batch > 448:
            return "requires head dimension 64/128 and at most 448 requests"
        page = call.page_size
        if page < 16 or page % 16 or not (128 % page == 0 or page % 128 == 0 or page == 48):
            return "requires a page size dividing 128, a multiple of 128, or 48"
        small = call.max_seqlen_q <= max(128, call.batch)
        if small and (call.batch == 1 or call.dim != 128 or call.heads // call.heads_kv > 16):
            return "retains the existing decode paths outside the small MMA region"
        if call.uses_sliding_window and not small:
            return "uses the general paged implementation for long windowed calls"
        if call.page_size * call.max_pages_per_req <= 1024:
            return "retains the packed path for short cache capacities"
        return None

    @classmethod
    def applies(cls, call):
        return cls.refusal(call) is None

    @classmethod
    def entry_for(cls, call):
        names = (
            "batch",
            "heads",
            "heads_kv",
            "dim",
            "page_size",
            "max_pages_per_req",
            "is_causal",
            "dtype",
            "sm_scale",
            "softcap",
            "window_size_left",
            "window_size_right",
        )
        args = {name: getattr(call, name) for name in names}
        index = call.device.index if call.device is not None else None
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        batch,
        heads,
        heads_kv,
        dim,
        page_size,
        max_pages_per_req,
        is_causal,
        dtype,
        sm_scale=None,
        softcap=0.0,
        window_size_left=-1,
        window_size_right=-1,
        config=None,
        tune=False,
        device_index=None,
    ):
        super().__init__(device_index=device_index)
        self.batch, self.heads, self.heads_kv = batch, heads, heads_kv
        self.dim, self.page_size, self.max_pages_per_req = dim, page_size, max_pages_per_req
        self.is_causal, self.dtype = is_causal, dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.window_size_left, self.window_size_right = window_size_left, window_size_right
        self._counters = {}
        self._sm_count = get_sm_count(device_index)
        self.init_config(config, tune)

    @property
    def default_config(self):
        group = self.heads // self.heads_kv
        capacity = self.page_size * self.max_pages_per_req
        # H200 measurements cover these head shapes. Keep the older settings
        # elsewhere until the dispatch cost model covers the remaining shapes.
        packed = self.dim == 128 and self.heads_kv == 8 and group in (4, 8, 16)
        return dict(
            decode_q=max(self.batch, min(128, max(32, 2 * self.batch)))
            if packed
            else max(128, self.batch),
            decode_splits=0,
            short_q=64 if packed else min(32, max(1, 128 // group)),
            splits=1 if packed else 4 if self.batch < 16 and capacity > 1024 else 1,
            load_mode="async_grouped_ahead" if self.page_size < 64 else "paged",
        )

    def forward(
        self,
        q,
        k_pool,
        v_pool,
        cache_seqlens,
        page_table,
        cu_seqlens_q=None,
        rope_cos=None,
        rope_sin=None,
    ):
        if q.shape[0] == 0:
            return torch.empty_like(q)
        c = self.config
        # Window masking is implemented by the decode specialization; its
        # applicability bound must remain consistent with refusal().
        windowed = self.window_size_left >= 0 or self.window_size_right >= 0
        decode = q.shape[0] <= (max(128, self.batch) if windowed else c["decode_q"])
        cut = c["short_q"] if self.batch > 1 else 0
        if decode:
            blocks = q.shape[0] * self.heads_kv
            need = (self._sm_count + blocks - 1) // blocks
            splits = c["decode_splits"] or min(32, 1 << (need - 1).bit_length())
        else:
            splits = c["splits"]
            # Even if every short request reaches the cut, these long-query
            # tiles remain. Enough independent work makes extra KV splits wasteful.
            long_tokens = max(0, q.shape[0] - self.batch * cut)
            long_blocks = ((long_tokens + 127) // 128) * self.heads
            if long_blocks >= self._sm_count:
                splits = 1
        program = paged_unified_kernel(
            self.batch,
            self.heads,
            self.heads_kv,
            self.dim,
            self.page_size,
            self.max_pages_per_req,
            self.is_causal,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            decode,
            splits,
            self._sm_count,
            cut,
            c["load_mode"],
            self.window_size_left,
            self.window_size_right,
        )
        rows = (
            16 if decode else max(128, ((cut * (self.heads // self.heads_kv) + 127) // 128) * 128)
        )
        owners = q.shape[0] if decode else self.batch
        count = owners * self.heads_kv if decode else 2 + owners * self.heads_kv * (rows // 64)
        if decode and splits == 1:
            # This specialization never accesses global scheduling state.
            counter = torch.empty(count, dtype=torch.int32, device=q.device)
        elif torch.cuda.is_current_stream_capturing():
            counter = _new_counters(count, q.device)
        else:
            key = (torch.cuda.current_stream(q.device).cuda_stream, decode, count)
            counter = self._counters.get(key)
            if counter is None:
                counter = _new_counters(count, q.device)
                self._counters[key] = counter
        lse = torch.empty(owners, self.heads_kv, splits, rows, dtype=torch.float32, device=q.device)
        part = torch.empty(*lse.shape, self.dim, dtype=q.dtype, device=q.device)
        out = torch.empty_like(q)
        program(q, k_pool, v_pool, cu_seqlens_q, cache_seqlens, page_table, out, counter, part, lse)
        return out
