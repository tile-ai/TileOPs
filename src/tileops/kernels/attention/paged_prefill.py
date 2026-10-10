"""The constructor, build identity and default tiling every paged GQA prefill implementation
shares."""

from typing import Optional

import torch

from tileops.kernels.attention.call_spec import AttentionCall, GQAPrefillPagedFwdInterface
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.utils import get_shared_memory_optin

__all__ = ["PagedPrefillKernel"]


class PagedPrefillKernel(Kernel, GQAPrefillPagedFwdInterface):
    """Base for every in-tree implementation of the paged GQA prefill interface."""

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None``.

        Every implementation indexes pages by shift and attends over the same query, K and V
        tiles. A subclass states its own region and asks this for the page-size and tile limits.
        """
        reason = cls.page_size_refusal(call.page_size)
        if reason is not None or not call.smem_budget:
            return reason
        need = cls._shared_bytes(cls._tilings(call.dim)[-1], call.dim, call.dtype.itemsize)
        if need <= call.smem_budget:
            return None
        return (
            f"its narrowest default tile needs {need} bytes of shared memory per block at head "
            f"dim {call.dim}; the device gives {call.smem_budget}"
        )

    @staticmethod
    def _tilings(dim: int) -> list[dict]:
        """The default's tilings, widest first. The query tile stays at 64 rows: each warp takes
        16, and where the K and V tiles narrow one block fills the SM, so fewer rows idle it."""
        wide = {"block_m": 64, "block_n": 64 if dim <= 128 else 32, "num_stages": 1, "threads": 128}
        return [wide, {**wide, "block_n": 16}]

    @staticmethod
    def _shared_bytes(config: dict, dim: int, elem: int) -> int:
        """Shared memory of the attend program at *config*: the query tile, one K and one V tile."""
        return (config["block_m"] + 2 * config["block_n"]) * dim * elem

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(self.device_index), self.dim, self.dtype.itemsize
        )

    @classmethod
    def _default_config_for(cls, budget: int, dim: int, elem: int) -> dict:
        """The widest tiling *budget* bytes of shared memory per block hold."""
        tilings = cls._tilings(dim)
        return next((c for c in tilings if cls._shared_bytes(c, dim, elem) <= budget), tilings[-1])

    @staticmethod
    def page_size_refusal(page_size: int) -> Optional[str]:
        """Why pages of *page_size* tokens cannot be indexed by shift; the builders ask too."""
        if page_size <= 0 or page_size & (page_size - 1) != 0:
            return "requires a power-of-two page_size"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The cache identity and the thunk that builds this class for *call*.

        Every implementation of the slot takes this constructor, so the base states
        it once. The device index is in the identity because the kernel is compiled
        for the architecture it is built on.
        """
        index = call.device.index if call.device is not None else None
        args = dict(
            batch=call.batch,
            heads=call.heads,
            heads_kv=call.heads_kv,
            max_pages_per_req=call.max_pages_per_req,
            page_size=call.page_size,
            dim=call.dim,
            is_causal=call.is_causal,
            dtype=call.dtype,
            sm_scale=call.sm_scale,
            softcap=call.softcap,
            max_position=call.max_position,
            rotary_dim=call.rotary_dim,
        )
        return (*args.values(), index), lambda: cls(**args, device_index=index)

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        max_pages_per_req: int,
        page_size: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        max_position: Optional[int] = None,
        rotary_dim: Optional[int] = None,
        config: Optional[dict] = None,
        tune: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.max_pages_per_req = max_pages_per_req
        self.page_size = page_size
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self._build_program()
        self.init_config(config, tune)

    def _build_program(self) -> None:
        """Build whatever the implementation launches beyond its wrapped call."""
