"""Paged gather inputs and the independent indexing reference."""

import torch

from workloads.attention.gqa.call_metadata import _dtype, _segments
from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device
from workloads.numerics import Exact
from workloads.sequence_metadata import make_cu_seqlens
from workloads.workload_base import CallWorkload, WorkloadBase


def paged_cache_gather_result(op, dst, *inputs):
    """Expose the write-only destination as the numeric result of an in-place gather."""
    op(dst, *inputs)
    return dst


class PagedKVCacheGatherWorkload(WorkloadBase):
    """Request ranges in a fragmented cache, with arbitrary trailing row dimensions."""

    def __init__(
        self,
        seq_lens,
        row_shape,
        page_size,
        dtype,
        out_dtype=None,
        starts=None,
        num_pages=None,
        table_width=None,
    ):
        self.seq_lens, self.row_shape = seq_lens, tuple(row_shape)
        self.page_size, self.dtype = page_size, dtype
        self.out_dtype, self.starts = out_dtype, starts
        ends = [n + s for n, s in zip(seq_lens, starts or [0] * len(seq_lens), strict=True)]
        self.table_width = table_width or max(1, -(-max(ends, default=0) // page_size))
        self.num_pages = num_pages or max(1, len(seq_lens) * self.table_width)

    def gen_inputs(self):
        device = run_device()
        shape = (self.num_pages, self.page_size, *self.row_shape)
        cache = torch.randn(
            shape, dtype=torch.float32, device=device, generator=self.rng("cache", device=device)
        ).to(self.dtype)
        dst = torch.full(
            (sum(self.seq_lens), *self.row_shape),
            float("nan"),
            dtype=self.out_dtype or self.dtype,
            device=device,
        )
        table = make_fragmented_block_table(len(self.seq_lens), self.table_width, self.num_pages)
        starts = (
            None
            if self.starts is None
            else torch.tensor(self.starts, dtype=torch.int32, device=device)
        )
        scale = (
            torch.tensor([1.25], dtype=torch.float32, device=device)
            if self.dtype == torch.float8_e4m3fn
            else None
        )
        return dst, cache, table, make_cu_seqlens(self.seq_lens), starts, scale

    def ref_program(self, dst, cache, block_table, cu_seq_lens, seq_starts=None, scale=None):
        """Index logical slots independently of the kernel's per-request tile scheduling."""
        offsets = cu_seq_lens.tolist()
        starts = seq_starts.tolist() if seq_starts is not None else [0] * (len(offsets) - 1)
        slots = []
        for b, (begin, end) in enumerate(zip(offsets, offsets[1:], strict=False)):
            positions = torch.arange(starts[b], starts[b] + end - begin, device=cache.device)
            slots.append(
                block_table[b, positions // cache.shape[1]].long() * cache.shape[1]
                + positions % cache.shape[1]
            )
        if dst.numel() == 0:
            return torch.empty_like(dst)
        rows = cache.flatten(0, 1)[torch.cat(slots)]
        if scale is not None:
            rows = rows.float() * scale
        return rows.to(dst.dtype)

    def verification(self, *inputs):
        # Copy and one FP32 multiply followed by conversion have no reduction-order error.
        return Exact(rtol=0, atol=0)


class PagedKVCacheGatherCall(CallWorkload, PagedKVCacheGatherWorkload):
    """The gather manifest call, including device-resident offsets and scale."""

    def __init__(self, call):
        CallWorkload.__init__(self, call)
        PagedKVCacheGatherWorkload.__init__(
            self,
            _segments(call.values("cu_seq_lens")),
            call.ix["E"],
            call.ix["PS"],
            _dtype(call, "cache"),
            out_dtype=getattr(torch, call.params["out_dtype"])
            if call.params["out_dtype"]
            else None,
            starts=call.values("seq_starts") if call.present("seq_starts") else None,
            num_pages=call.ix["NP"],
            table_width=call.ix["W"],
        )

    gen_inputs = PagedKVCacheGatherWorkload.gen_inputs
