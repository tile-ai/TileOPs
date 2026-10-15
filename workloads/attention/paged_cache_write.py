"""Inputs and the independent reference for the paged KV cache write."""

import torch

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "PagedKVCacheWriteCall",
    "PagedKVCacheWriteWorkload",
    "paged_cache_write_result",
]


def paged_cache_write_result(subject, *inputs):
    """Expose the two declared cache writes as the numeric result of an in-place op."""
    subject(*inputs)
    return inputs[2], inputs[3]


class PagedKVCacheWriteWorkload(WorkloadBase):
    """New rows scattered into a paged cache by slot, the pool's other rows left alone.

    ``slots[n]`` is the pool row token ``n`` writes, or ``-1`` for a token that writes
    nothing. Omitted, the slots are distinct random rows, which is what a serving step
    hands over; a caller passing its own list exercises the skip path, a slot at either
    end of a page, and slots out of order.
    """

    def __init__(
        self,
        tokens: int,
        heads_kv: int,
        dim: int,
        page_size: int,
        num_pages: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype | None = None,
        slots: list[int] | None = None,
    ) -> None:
        self.tokens, self.heads_kv, self.dim = tokens, heads_kv, dim
        self.page_size, self.num_pages = page_size, num_pages
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.slots = slots

    @property
    def row_width(self) -> int:
        return self.heads_kv * self.dim

    @property
    def physical_tokens(self) -> int:
        return self.num_pages * self.page_size

    @property
    def quantized(self) -> bool:
        return self.cache_dtype == torch.float8_e4m3fn

    def gen_inputs(self):
        if self.quantized and self.tokens > 0 and self.physical_tokens <= 0:
            raise ValueError("a quantized cache needs at least one row")
        device = run_device()
        row = (self.heads_kv, self.dim)
        k = torch.randn(
            (self.tokens, *row),
            dtype=torch.float32,
            device=device,
            generator=self.rng("k", device=device),
        ).to(self.dtype)
        v = torch.randn(
            (self.tokens, *row),
            dtype=torch.float32,
            device=device,
            generator=self.rng("v", device=device),
        ).to(self.dtype)
        pool = torch.randn(
            (self.num_pages, self.page_size, *row),
            dtype=torch.float32,
            device=device,
            generator=self.rng("pages", device=device),
        )
        # The two pools differ, so a kernel writing one row to both cannot pass.
        k_pages = pool.to(self.cache_dtype)
        v_pages = (-pool).to(self.cache_dtype)
        if self.slots is None:
            if self.tokens > self.physical_tokens:
                raise ValueError(
                    f"{self.tokens} tokens need {self.tokens} distinct slots but the pool "
                    f"holds {self.physical_tokens} rows"
                )
            drawn = torch.randperm(self.physical_tokens, generator=self.rng("slots"))[: self.tokens]
            slots = drawn.to(device=device, dtype=torch.int64)
        else:
            if len(self.slots) != self.tokens:
                raise ValueError(f"slots holds {len(self.slots)} rows for {self.tokens} tokens")
            slots = torch.tensor(self.slots, dtype=torch.int64, device=device)
        if self.quantized:
            # Distinct, non-unit divisors: a kernel that drops one, or swaps the two,
            # stores something the reference does not.
            k_scale = torch.tensor([1.25], dtype=torch.float32, device=device)
            v_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
        else:
            k_scale = v_scale = None
        return k, v, k_pages, v_pages, slots, k_scale, v_scale

    def ref_program(self, k, v, k_pages, v_pages, slot_mapping, k_scale=None, v_scale=None):
        """Scatter by index, with saturating FP8 conversion that preserves NaN.

        The pages are cloned before they are written: the reference runs on the shared
        inputs, and a call that wrote them would leave nothing to compare against. A
        token whose slot is ``-1`` writes nothing, and a row no token names keeps the
        value it was handed, which the caller's comparison reads back.
        """
        k_pages, v_pages = k_pages.clone(), v_pages.clone()
        written = (slot_mapping >= 0).nonzero().squeeze(1)
        rows = slot_mapping[written]
        for pages, new, scale in (
            (k_pages, k, k_scale),
            (v_pages, v, v_scale),
        ):
            flat = pages.view(-1, *pages.shape[2:])
            stored = new[written]
            if scale is not None:
                # Express saturation explicitly: some torch FP8 casts turn overflow
                # into NaN. torch.clamp preserves NaN while bounding other values.
                limit = torch.finfo(pages.dtype).max
                stored = (stored.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
            flat[rows] = stored
        return k_pages, v_pages

    def verification(self, *inputs):
        # The stored bytes are the whole result: a named row is written exactly and a row
        # no slot names must still hold what it held, so no tolerance is owed either way.
        return Exact(rtol=0, atol=0)


class PagedKVCacheWriteCall(CallWorkload, PagedKVCacheWriteWorkload):
    """The write manifest call, including the device-resident slots."""

    def __init__(self, call):
        CallWorkload.__init__(self, call)
        PagedKVCacheWriteWorkload.__init__(
            self,
            tokens=call.indices["N"],
            heads_kv=call.indices["H_kv"],
            dim=call.indices["D"],
            page_size=call.indices["PS"],
            num_pages=call.indices["NP"],
            dtype=getattr(torch, call.tensors["k"][1]),
            cache_dtype=getattr(torch, call.tensors["k_pages"][1]),
            slots=call.metadata_values("slot_mapping"),
        )

    gen_inputs = PagedKVCacheWriteWorkload.gen_inputs

    def arguments(self):
        """The op is built from no tensors, so a case needs none to construct it.

        The base materializes the call to hand a construction-time tensor over; this entry
        declares none, and materializing would allocate both page pools for nothing.
        """
        return self.call.arguments({})
