"""
GLA (Gated Linear Attention) decode (single-step recurrence).

    S_new = diag(exp(gk)) @ S + outer(k, v)
    o     = scale * q^T @ S_new

where gk is per-key-dimension log-space gate [B, H, DK].

The fp16/bf16 kernel runs one block a ``(batch, head, value slice)`` and reads each state
element once; the fp32 kernel runs one block a head.
"""

import functools
import math
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import LOG2E
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.linear_attention.call_spec import (
    GLADecodeCall,
    GLADecodeFwdInterface,
    head_count_refusal,
)
from tileops.utils import get_sm_count

__all__ = ["GLADecodeFP32Kernel", "GLADecodeKernel"]


@functools.lru_cache(maxsize=32)
def _gla_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    block_v: int,
    threads: int,
    dtype: str,
    scale: float,
):
    """Build the decode program for one value-slice width.

    Args:
        batch: Sequences in the call.
        head: Heads a sequence.
        dim_k: The key dimension, reduced inside a block.
        dim_v: The value dimension, split into ``block_v``-wide slices, one a block.
        block_v: The value columns one block owns.
        threads: Upper bound on the threads a block.
        dtype: TileLang name of the input and output dtype.
        scale: Query scale.

    Returns:
        The built TileLang program.
    """
    accum_dtype = "float32"
    if dim_v % block_v != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by block_v={block_v}")
    value_blocks = dim_v // block_v
    # The key-axis exchange pairs threads by XOR, so both group counts must be powers of two;
    # otherwise each thread holds the whole slice width.
    vector = math.gcd(block_v, 16 // getattr(torch, dtype).itemsize)
    if (block_v // vector) & (block_v // vector - 1):
        vector = block_v
    column_groups = block_v // vector
    row_groups = math.gcd(dim_k & -dim_k, max(threads // column_groups, 1))
    block_threads = row_groups * column_groups

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def decode():
        state_layout = T.Fragment(
            [dim_k, block_v],
            forward_thread_fn=lambda i, j: (i % row_groups) * column_groups + j // vector,
            forward_index_fn=lambda i, j: (i // row_groups) * vector + j % vector,
        )

        @T.prim_func
        def gla_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(value_blocks, head, batch, threads=block_threads) as (vb, hid, bid):
                v0 = vb * block_v
                h_in = T.alloc_fragment([dim_k, block_v], dtype)
                h_out = T.alloc_fragment([dim_k, block_v], dtype)
                h_new = T.alloc_fragment([dim_k, block_v], accum_dtype)
                o_frag = T.alloc_fragment([block_v], accum_dtype)
                T.annotate_layout({h_in: state_layout, h_out: state_layout, h_new: state_layout})
                q_shared = T.alloc_shared([dim_k], accum_dtype)
                k_shared = T.alloc_shared([dim_k], accum_dtype)
                decay_shared = T.alloc_shared([dim_k], accum_dtype)
                v_shared = T.alloc_shared([block_v], accum_dtype)

                T.copy(state[bid, hid, 0, v0], h_in)
                for i in T.Parallel(dim_k):
                    q_shared[i] = T.cast(q[bid, hid, i], accum_dtype)
                    k_shared[i] = T.cast(k[bid, hid, i], accum_dtype)
                    decay_shared[i] = T.exp2(T.cast(gk[bid, hid, i], accum_dtype) * LOG2E)
                for j in T.Parallel(block_v):
                    v_shared[j] = T.cast(v[bid, hid, v0 + j], accum_dtype)

                for i, j in T.Parallel(dim_k, block_v):
                    h_new[i, j] = (
                        decay_shared[i] * T.cast(h_in[i, j], accum_dtype)
                        + k_shared[i] * v_shared[j]
                    )
                for i, j in T.Parallel(dim_k, block_v):
                    h_out[i, j] = T.cast(h_new[i, j], dtype)
                T.copy(h_out, new_state[bid, hid, 0, v0])

                for i, j in T.Parallel(dim_k, block_v):
                    h_new[i, j] = q_shared[i] * h_new[i, j]
                T.reduce_sum(h_new, o_frag, dim=0, batch=vector)
                for j in T.Parallel(block_v):
                    o[bid, hid, v0 + j] = T.cast(scale * o_frag[j], dtype)

        return gla_decode

    return decode()


def _decode_entry(cls: type, call: GLADecodeCall) -> Entry:
    """The entry for a GLA decode kernel: both take the same construction arguments.

    The device index is in the identity because the kernel is compiled for the
    architecture it is built on.
    """
    index = call.device.index if call.device is not None else None
    dtype = Kernel.dtype_to_str(call.dtype)
    identity = (call.batch, call.heads, call.dim_k, call.dim_v, call.scale, dtype, index)
    return identity, lambda: cls(
        call.batch,
        call.heads,
        call.dim_k,
        call.dim_v,
        scale=call.scale,
        dtype=dtype,
        device_index=index,
    )


class GLADecodeKernel(Kernel, GLADecodeFwdInterface):
    """GLA single-step decode kernel, one block a ``(batch, head, value slice)``, fp32 accumulation."""

    supported_archs: list[int] = [80, 89, 90]
    general = True

    @classmethod
    def refusal(cls, call: GLADecodeCall) -> Optional[str]:
        return head_count_refusal(call.heads)

    @classmethod
    def entry_for(cls, call: GLADecodeCall) -> Entry:
        return _decode_entry(cls, call)

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: str = "float32",
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale if scale > 0 else dim_k**-0.5
        self.dtype = dtype

        self.init_config(config)
        self._build_program()

    def _build_program(self) -> None:
        """Compile the decode program the current config states."""
        self._kernel_fn = _gla_decode_tl(
            self.batch,
            self.head,
            self.dim_k,
            self.dim_v,
            self.config["block_v"],
            self.config["threads"],
            self.dtype_str,
            self.scale,
        )

    def _block_v_options(self) -> list[int]:
        """The value-slice widths this shape builds, narrowest first."""
        return [b for b in (8, 16, 32, 64, 128) if b < self.dim_v and self.dim_v % b == 0] + [
            self.dim_v
        ]

    def autotune(self, warmup: int = 10, rep: int = 20) -> None:
        """Sweep block_v and threads, then rebuild the program."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config

        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v
        torch_dtype = getattr(torch, self.dtype_str)
        q = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch_dtype)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch_dtype)
        gk = -torch.rand(B, H, DK, device="cuda", dtype=torch_dtype)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch_dtype)

        print(f"Start autotuning {self.__class__.__name__}...")
        candidates = self.tuning_candidates(
            [
                {"block_v": block_v, "threads": threads}
                for block_v in self._block_v_options()
                for threads in (128, 256)
            ]
        )
        for config in candidates:
            try:
                fn = _gla_decode_tl(
                    B, H, DK, DV, config["block_v"], config["threads"], self.dtype_str, self.scale
                )
                t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state), warmup=warmup, rep=rep)
                if t < best_time:
                    best_time = t
                    best_config = config
            except Exception:
                continue

        self.config = best_config
        print(f"Best config: {self.config}")
        self._build_program()

    @property
    def default_config(self) -> dict:
        """The narrowest value slice whose blocks fit one wave, and 128 threads."""
        sm_count = get_sm_count(self.device_index)
        sequences = self.batch * self.head
        options = self._block_v_options()
        block_v = next(
            (b for b in options if sequences * (self.dim_v // b) <= sm_count), options[-1]
        )
        return {"block_v": block_v, "threads": 128}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, gk, state)


# FP32-precision decode kernel (no T.gemm → avoids TF32 mantissa truncation)


@functools.lru_cache(maxsize=32)
def _gla_decode_fp32_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    k_tile: int,
    scale: float = -1.0,
):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@q_gated via scalar accumulation in full fp32.
    """
    dtype = "float32"
    accum_dtype = "float32"
    if dim_k % k_tile != 0:
        raise ValueError(f"dim_k={dim_k} must be divisible by k_tile={k_tile}")
    # FIXME(staged-rollout): a state that fits one tile is loaded without TMA
    #
    # Broken invariant: every k_tile dividing dim_k builds a program that runs.
    # Why: with k_tile == dim_k the pipelined state loop runs once, and TileLang lowers its
    #      copy to a 1D bulk copy whose producer never arrives at the barrier under SM90
    #      warp specialization, so the first launch hangs.
    # Cleanup: drop disable_tma once TileOPs pins TileLang 0.1.13 or later, which carries
    #          tile-ai/tilelang#2737.
    load_state_without_tma = k_tile == dim_k

    if scale <= 0:
        scale = dim_k**-0.5

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3"],
    )
    def _decode_func(num_stages, threads=128):
        @T.prim_func
        def gla_decode_fp32(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            gk: T.Tensor([batch, head, dim_k], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                # Fragment accumulator for S @ q_gated (full fp32, no TF32)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                # Preload q, k, v, gk into shared memory to avoid global access
                # in pipelined loop (fixes TileLang warp-specialization issue)
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                gk_shared = T.alloc_shared([dim_k], dtype)

                # Preload tensors into shared memory
                for i in T.Parallel(dim_k):
                    q_shared[i] = q[bid, hid, i]
                    k_shared[i] = k[bid, hid, i]
                    gk_shared[i] = gk[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_shared[i] = v[bid, hid, i]

                T.fill(sq_frag, 0.0)

                # === Pass 1: Element-wise matvec (full fp32 precision) ===
                # S @ q_gated where q_gated = q * exp(gk)
                for kk in T.Serial(dim_k):
                    gk_val = gk_shared[kk]
                    alpha_i = T.exp2(gk_val * LOG2E)
                    q_gated = q_shared[kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]

                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]

                # o = scale * (S @ q_gated) + scale * (q . k) * v
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j]

                # === Pass 2: State update with async prefetch ===
                # new_state[dk, dv] = exp(gk[dk]) * state[dk, dv] + k[dk] * v[dv]
                # NOTE: No intermediate variables inside T.Parallel to avoid BindNode issue
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(
                        state[bid, hid, kt * k_tile, 0], h_tile, disable_tma=load_state_without_tma
                    )
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = (
                            T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * h_tile[kk, j]
                            + k_shared[kt * k_tile + kk] * v_shared[j]
                        )

        return gla_decode_fp32

    return _decode_func


class GLADecodeFP32Kernel(Kernel, GLADecodeFwdInterface):
    """FP32-precision GLA decode kernel (no TF32 tensor cores).

    Uses element-wise matvec instead of T.gemm to avoid TF32 mantissa
    truncation that causes ~1e-3 error per step, compounding over multi-step
    decode.  Intended for fp32 dtype only.
    """

    supported_archs: list[int] = [80, 89, 90]
    # The key-dimension tile the untuned decode program splits the state into.
    _DEFAULT_K_TILE = 16

    @classmethod
    def refusal(cls, call: GLADecodeCall) -> "str | None":
        if call.dtype != torch.float32:
            return f"serves float32, got {call.dtype}"
        return head_count_refusal(call.heads) or super().refusal(call)

    @classmethod
    def entry_for(cls, call: GLADecodeCall) -> Entry:
        return _decode_entry(cls, call)

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: str = "float32",
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        if dtype != "float32":
            raise ValueError(f"{self.__class__.__name__} only supports float32")
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale if scale > 0 else dim_k**-0.5

        self.init_config(config)
        self._build_program()

    def _build_program(self) -> None:
        """Compile the fp32 decode program the current config states."""
        self._kernel_fn = _gla_decode_fp32_tl(
            self.batch,
            self.head,
            self.dim_k,
            self.dim_v,
            self.config["k_tile"],
            self.scale,
        )(self.config["num_stages"], self.config["threads"])

    def autotune(self, warmup: int = 10, rep: int = 20) -> None:
        """Sweep k_tile, num_stages and threads, then rebuild the program."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config
        B, H, DK, DV = self.batch, self.head, self.dim_k, self.dim_v

        q = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, DK, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, DV, device="cuda", dtype=torch.float32)
        gk = -torch.rand(B, H, DK, device="cuda", dtype=torch.float32)
        state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch.float32)

        print(f"Start autotuning {self.__class__.__name__}...")
        candidates = self.tuning_candidates(
            [
                {"num_stages": num_stages, "threads": threads, "k_tile": k_tile}
                for k_tile in (16, 32, 64)
                if DK % k_tile == 0
                for num_stages in (1, 2, 3)
                for threads in (128, 256)
            ]
        )
        for config in candidates:
            try:
                fn = _gla_decode_fp32_tl(
                    B,
                    H,
                    DK,
                    DV,
                    config["k_tile"],
                    self.scale,
                )(config["num_stages"], config["threads"])
                t = do_bench(lambda _fn=fn: _fn(q, k, v, gk, state), warmup=warmup, rep=rep)
                if t < best_time:
                    best_time = t
                    best_config = config
            except Exception:
                continue

        self.config = best_config
        print(f"Best config: {self.config}")
        self._build_program()

    @property
    def default_config(self) -> dict:
        return {
            "num_stages": 2,
            "threads": 128,
            "k_tile": self._DEFAULT_K_TILE,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, gk, state)
