"""Shared construction contract for packed variable-length attention kernels."""

from typing import Callable, Optional

import torch

from tileops.kernels.attention.call_spec import AttentionCall, GQAVarlenFwdInterface
from tileops.kernels.attention.varlen_rope import VarlenKeyRoPE
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["VarlenKernel"]


class VarlenKernel(Kernel, GQAVarlenFwdInterface):
    """Facts shared by in-tree kernels serving the Varlen Op."""

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """One object per call-independent fact set; it reads the packed totals from Q/K on
        every forward, and its program factory caches the specialization each needs."""
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
            fuse_rope=call.fuse_rope,
            max_position=call.max_position if call.max_position is not None else 1,
            rotary_dim=call.rotary_dim if call.rotary_dim else call.dim,
            rope_layout=call.rope_layout,
            device_index=call.device.index if call.device is not None else None,
        )
        return tuple(args.values()), lambda: cls(**args)

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
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        accum_dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
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
        self.fuse_rope = fuse_rope
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        # A rotated key row is read by every query tile of its request and by every query
        # head of its KV group, so it is rotated once here rather than inside the scan.
        self.key_rope = (
            VarlenKeyRoPE(
                batch,
                heads_kv,
                dim,
                max_position,
                rotary_dim,
                rope_layout,
                self.rotated_dtype_str,
                self.rope_table_dtype_str,
            )  # fmt: skip
            if fuse_rope
            else None
        )
        self.kernel = self._make_kernel()
        self._supply_prog = self._make_supply_prog()
        self.init_config(config)

    @property
    def rotated_dtype_str(self) -> str:
        """Element type of the Q and K the rotation reads and writes.

        It is this kernel's element type unless a subclass attends over narrower inputs
        than it emits, as an FP8 kernel does.
        """
        return self.dtype_str

    @property
    def rope_table_dtype_str(self) -> str:
        """Element type of the ``cos`` and ``sin`` tables, which the manifest ties to the
        output rather than to the rotated tensor."""
        return self.dtype_str

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
        rope_rows = self.max_position
        rope_half = self.rotary_dim // 2
        expected = 7 if self.fuse_rope else 5

        def supply_prog(params):
            if len(params) != expected:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects q, k, v, two "
                    f"cumulative-length inputs and the RoPE tables it was built with, "
                    f"got {len(params)} parameters"
                )
            device = get_current_device()
            cu_seqlens = torch.arange(
                0,
                total + 1,
                tokens_per_request,
                dtype=torch.int32,
                device=device,
            )
            supplied = [
                torch.randn(total, heads, dim, dtype=dtype, device=device),
                torch.randn(total, heads_kv, dim, dtype=dtype, device=device),
                torch.randn(total, heads_kv, dim, dtype=dtype, device=device),
                cu_seqlens,
                cu_seqlens.clone(),
            ]
            if expected == 7:
                angles = torch.randn(rope_rows, rope_half, device=device)
                supplied += [angles.cos().to(dtype), angles.sin().to(dtype)]
            return supplied

        return supply_prog

    @property
    def autotune_supply_prog(self) -> Callable:
        return self._supply_prog

    @property
    def accum_dtype_str(self) -> str:
        return "float" if self.accum_dtype == torch.float32 else "double"
