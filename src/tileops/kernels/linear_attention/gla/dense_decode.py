"""Dense GLA decode with caller-owned FP32 recurrent state.

One block owns a warp-wide column tile of one ``(batch, head)`` state and advances it by
one token; a lane group within the block splits the key dimension for one column.

``has_initial_state`` is a build flag. A build without a starting state reads neither the
state slice nor the gate, because every term the gate scales is zero, and the step is the
outer product of the token's key and value alone.
"""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import head_count_refusal
from tileops.kernels.linear_attention.gla.call_spec import (
    GLAInferenceCallSpec,
    GLAInferenceFwdInterface,
    build_entry,
    serves_dense,
)
from tileops.utils import WARP_LANES

__all__ = ["GLADenseDecodeFwdKernel"]


@functools.lru_cache(maxsize=32)
def _gla_dense_decode_tl(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    scale: float,
    has_initial_state: bool,
    lane_group: int,
):
    """Build the decode program for one state presence and block shape.

    Args:
        batch: Sequences in the call.
        heads: Query, key, value and state heads.
        dim_k: The key dimension, split across the block's lane groups.
        dim_v: The value dimension, split into warp-wide column tiles.
        dtype: TileLang name of the token dtype.
        scale: Query scale.
        has_initial_state: Read the caller's state and decay it by the gate; otherwise
            start the step from zero and read no gate.
        lane_group: Lanes that split the key dimension for one state column.

    Returns:
        The built TileLang program.
    """
    v_tile = WARP_LANES
    threads = v_tile * lane_group
    value_tiles = dim_v // v_tile
    total_blocks = batch * heads * value_tiles
    k_chunk = dim_k // lane_group
    lane_group_stages = lane_group.bit_length() - 1

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16", "--use_fast_math"],
    )
    def decode():
        # A parameter this build leaves unread is declared one element wide, and the kernel
        # hands it a one-element placeholder.
        unread = [1]
        gate_shape = [batch, 1, heads, dim_k] if has_initial_state else unread
        state_shape = [batch, heads, dim_k, dim_v] if has_initial_state else unread

        @T.prim_func
        def main(
            q: T.Tensor([batch, 1, heads, dim_k], dtype),
            k: T.Tensor([batch, 1, heads, dim_k], dtype),
            v: T.Tensor([batch, 1, heads, dim_v], dtype),
            g: T.Tensor(gate_shape, dtype),
            initial_state: T.Tensor(state_shape, "float32"),
            o: T.Tensor([batch, 1, heads, dim_v], dtype),
            final_state: T.Tensor([batch, heads, dim_k, dim_v], "float32"),
        ):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                value_tile = block % value_tiles
                batch_head = block // value_tiles
                bid = batch_head // heads
                hid = batch_head - bid * heads
                value_lane = tx // lane_group
                k_rank = tx - value_lane * lane_group
                value_idx = value_tile * v_tile + value_lane
                k_begin = k_rank * k_chunk

                q_shared = T.alloc_shared([dim_k], "float32")
                k_shared = T.alloc_shared([dim_k], "float32")
                decay_shared = T.alloc_shared([dim_k], "float32")
                acc = T.alloc_var("float32", init=0.0)

                for i in T.serial(T.ceildiv(dim_k, threads)):
                    idx = tx + i * threads
                    if idx < dim_k:
                        q_shared[idx] = T.cast(q[bid, 0, hid, idx], "float32") * scale
                        k_shared[idx] = T.cast(k[bid, 0, hid, idx], "float32")
                        if has_initial_state:
                            decay_shared[idx] = T.exp2(
                                T.cast(g[bid, 0, hid, idx], "float32") * LOG2E
                            )
                T.sync_threads()

                value = T.cast(v[bid, 0, hid, value_idx], "float32")
                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    step = k_shared[kk] * value
                    if has_initial_state:
                        state = decay_shared[kk] * initial_state[bid, hid, kk, value_idx] + step
                    else:
                        state = step
                    final_state[bid, hid, kk, value_idx] = state
                    acc += q_shared[kk] * state

                for stage in T.unroll(lane_group_stages):
                    acc += T.shfl_down(acc, 1 << stage, width=lane_group)
                if k_rank == 0:
                    o[bid, 0, hid, value_idx] = T.cast(acc, dtype)

        return main

    return decode()


class GLADenseDecodeFwdKernel(Kernel, GLAInferenceFwdInterface):
    """Fuse one GLA recurrence step and output projection in one state pass."""

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: GLAInferenceCallSpec) -> Optional[str]:
        return head_count_refusal(call.heads) or super().refusal(call)

    @classmethod
    def applies(cls, call: GLAInferenceCallSpec) -> bool:
        return serves_dense(call) and call.seq_len == 1

    @classmethod
    def entry_for(cls, call: GLAInferenceCallSpec) -> Entry:
        return build_entry(
            cls,
            call,
            batch=call.batch,
            heads=call.heads,
            dim_k=call.dim_k,
            dim_v=call.dim_v,
            has_initial_state=call.has_initial_state,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        scale: float,
        dtype: torch.dtype,
        has_initial_state: bool = False,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        if dim_k not in (64, 128) or dim_v not in (64, 128):
            raise ValueError("GLA dense decode requires K and V dimensions of 64 or 128")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("GLA dense decode requires float16 or bfloat16 activations")
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale
        self.dtype = dtype
        self.has_initial_state = has_initial_state
        self.init_config()
        self._kernel = _gla_dense_decode_tl(
            batch,
            heads,
            dim_k,
            dim_v,
            self.dtype_to_str(dtype),
            scale,
            has_initial_state,
            self.config["lane_group"],
        )
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        self._unread_gate = torch.empty(1, dtype=dtype, device=device)
        self._unread_state = torch.empty(1, dtype=torch.float32, device=device)

    @property
    def default_config(self) -> dict:
        """The lanes that split the key dimension for one state column.

        The state slice is a dependent global read, and a second lane group is what
        overlaps it; a build that reads no state has nothing to overlap and pays for the
        extra warp. Re-fit by sweeping ``lane_group`` against the decode workload rows of
        `benchmarks/ops/bench_gla.py`.
        """
        return {"lane_group": 2 if self.has_initial_state else 1}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_cpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None or cu_seqlens_cpu is not None:
            raise ValueError("GLA dense decode does not support packed varlen inputs")
        if self.has_initial_state and initial_state is None:
            raise ValueError("the build reads initial_state, but the call passed none")
        if self.has_initial_state:
            return self._kernel(q, k, v, g, initial_state)
        return self._kernel(q, k, v, self._unread_gate, self._unread_state)
