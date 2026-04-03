"""Shared Expert MLP Kernel — TileLang implementation."""

import functools
import math

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm.gemm import GemmKernel
from tileops.kernels.kernel import Kernel

__all__ = ["SharedExpertMLPKernel"]

_DEFAULT_CONFIG = {"block_m": 128, "block_n": 256, "block_k": 64, "num_stages": 3, "threads": 256, "enable_rasteration": True}


@functools.lru_cache(maxsize=16)
def _silu_mul_split_kernel(M: int, N: int, dtype_str: str):
    """Fused SiLU + elementwise multiply for split inputs: out = silu(gate) * up.

    Takes two separate [M, N] tensors (gate, up) as input.
    Uses FP32 accumulation for numerical stability.
    """
    dtype = dtype_str

    @tilelang.jit(
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, threads):
        gate_shape = (M, N)
        up_shape = (M, N)
        out_shape = (M, N)

        @T.prim_func
        def _main(
            gate: T.Tensor(gate_shape, dtype),
            up: T.Tensor(up_shape, dtype),
            out: T.Tensor(out_shape, dtype),
        ):
            num_blocks_m = math.ceil(M / block_m)
            num_blocks_n = math.ceil(N / block_n)
            with T.Kernel(num_blocks_m * num_blocks_n, threads=threads) as (bx,):
                bx_m = bx // num_blocks_n
                bx_n = bx % num_blocks_n
                m_start = bx_m * block_m
                n_start = bx_n * block_n

                for i, j in T.Parallel(block_m, block_n):
                    if m_start + i < M and n_start + j < N:
                        g = gate[m_start + i, n_start + j]
                        u = up[m_start + i, n_start + j]
                        g_f32 = T.cast(g, "float32")
                        u_f32 = T.cast(u, "float32")
                        sigmoid_g = T.sigmoid(g_f32)
                        result = g_f32 * sigmoid_g * u_f32
                        out[m_start + i, n_start + j] = T.cast(result, dtype)

        return _main

    return _func


class SharedExpertMLPKernel(Kernel):
    """Shared expert MLP: gate_up GEMM + SiLU + down GEMM.

    Uses GemmKernel (trans_b=True) for both GEMMs.

    Forward signature:
        hidden:    [T, H]
        w_gate_up: [2F, H]  — gate and up weights concatenated along dim 0
        w_down:    [H, F]
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(self, num_tokens: int, hidden_size: int, ffn_size: int,
                 dtype: torch.dtype = torch.bfloat16, config=None, tune: bool = False):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.init_config(config, tune)

        self._gemm_gate_up = GemmKernel(
            m=num_tokens, n=ffn_size * 2, k=hidden_size,
            dtype=dtype, trans_b=True, config=self.config,
        )
        self._gemm_down = GemmKernel(
            m=num_tokens, n=hidden_size, k=ffn_size,
            dtype=dtype, trans_b=True, config=self.config,
        )

    @property
    def default_config(self) -> dict:
        return dict(_DEFAULT_CONFIG)

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(self, hidden: torch.Tensor, w_gate_up: torch.Tensor,
                w_down: torch.Tensor) -> torch.Tensor:
        T_dim = self.num_tokens
        F = self.ffn_size

        # [T, H] @ [2F, H]^T -> [T, 2F]
        gate_up_out = self._gemm_gate_up(hidden, w_gate_up)

        # Split: .contiguous() is a no-op when T=1 (stride[0] stays 2F), use reshape instead
        gate = gate_up_out[:, :F].reshape(T_dim, F).contiguous()
        up = gate_up_out[:, F:].reshape(T_dim, F).contiguous()

        # SiLU + Mul in FP32
        silu_mul_fn = _silu_mul_split_kernel(T_dim, F, self.dtype_str)(
            self.config["block_m"], self.config["block_n"], self.config["threads"])
        gate_up = silu_mul_fn(gate, up)

        # [T, F] @ [H, F]^T -> [T, H]
        return self._gemm_down(gate_up, w_down)
