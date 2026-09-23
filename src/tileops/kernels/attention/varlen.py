"""Shared construction contract for packed variable-length attention kernels."""

from typing import Callable, Optional

import torch

from ..kernel_base import Entry, Kernel
from .call_spec import AttentionCall

__all__ = ["VarlenKernel", "varlen_entry"]


def _device_index(call: AttentionCall) -> Optional[int]:
    return call.device.index if call.device is not None else None


def varlen_entry(cls: type, call: AttentionCall) -> Entry:
    """Build one reusable Varlen kernel object from call-independent facts.

    Packed totals are deliberately absent. The object reads them from Q/K on
    every forward call; the TileLang program factory then caches the compiled
    specialization it needs.
    """
    args = dict(
        batch=call.batch,
        heads=call.heads,
        heads_kv=call.heads_kv,
        dim=call.dim,
        is_causal=call.is_causal,
        dtype=call.dtype,
        sm_scale=call.sm_scale,
        softcap=call.softcap,
        window_size_left=call.window_size_left,
        window_size_right=call.window_size_right,
        accum_dtype=call.accum_dtype,
        sm_count=call.sm_count,
        device_index=_device_index(call),
        tune=call.tune,
    )
    return tuple(args.values()), lambda: cls(**args)


class VarlenKernel(Kernel):
    """Facts shared by in-tree kernels serving the Varlen Op."""

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        accum_dtype: torch.dtype = torch.float32,
        sm_count: int = 0,
        config: Optional[dict] = None,
        tune: bool = False,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.accum_dtype = accum_dtype
        self.sm_count = sm_count
        self.kernel = self._make_kernel()
        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_kernel(self) -> Callable:
        """Return the dynamic TileLang program factory owned by this kernel."""
        raise NotImplementedError

    def _make_supply_prog(self) -> Callable:
        """Supply valid packed tensors and cumulative lengths while autotuning."""
        from tilelang.utils.device import get_current_device

        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        dim = self.dim
        dtype = self.dtype
        # Two 128-token tiles exercise the tiled loop while keeping every
        # candidate probe bounded. This is a synthetic tuning point, not a
        # claim that one config is optimal for every packed total.
        tokens_per_request = 256
        total = batch * tokens_per_request

        def supply_prog(params):
            if len(params) != 5:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects q, k, v and two "
                    f"cumulative-length inputs, got {len(params)} parameters"
                )
            device = get_current_device()
            cu_seqlens = torch.arange(
                0,
                total + 1,
                tokens_per_request,
                dtype=torch.int32,
                device=device,
            )
            return [
                torch.randn(total, heads, dim, dtype=dtype, device=device),
                torch.randn(total, heads_kv, dim, dtype=dtype, device=device),
                torch.randn(total, heads_kv, dim, dtype=dtype, device=device),
                cu_seqlens,
                cu_seqlens.clone(),
            ]

        return supply_prog

    @property
    def autotune_supply_prog(self) -> Callable:
        return self._supply_prog

    @property
    def accum_dtype_str(self) -> str:
        return "float" if self.accum_dtype == torch.float32 else "double"
