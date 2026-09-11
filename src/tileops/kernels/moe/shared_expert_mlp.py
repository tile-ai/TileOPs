"""Shared Expert MLP Kernel — TileLang implementation."""

import functools
import math

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.gemm.dense import GemmBasicKernel, GemmKernel
from tileops.kernels.grouped_gemm.heuristics import GemmType
from tileops.kernels.grouped_gemm.template import GemmTemplate
from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

__all__ = ["SharedExpertMLPKernel"]


@functools.lru_cache(maxsize=16)
def _silu_mul_fused_kernel(M: int, N: int, dtype_str: str):
    """Map ``gate_up[M, 2N]`` to ``silu(gate) * up[M, N]``."""
    dtype = dtype_str

    @tilelang.jit(
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _func(block_m, block_n, threads):
        gate_up_shape = (M, N * 2)
        out_shape = (M, N)

        @T.prim_func
        def _main(
            gate_up: T.Tensor(gate_up_shape, dtype),
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
                        g = gate_up[m_start + i, n_start + j]
                        u = gate_up[m_start + i, n_start + j + N]
                        g_f32 = T.cast(g, "float32")
                        u_f32 = T.cast(u, "float32")
                        sigmoid_g = T.sigmoid(g_f32)
                        result = g_f32 * sigmoid_g * u_f32
                        out[m_start + i, n_start + j] = T.cast(result, dtype)

        return _main

    return _func


class SharedExpertMLPKernel(Kernel):
    """Shared expert MLP producing ``[T, H]``.

    Inputs are ``hidden[T, H]``, concatenated ``w_gate_up[2F, H]``, and
    ``w_down[H, F]``. Hopper uses the dense template for wide shared experts
    above ``template_min_m``; other calls use the existing dense implementations.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config=None,
        tune: bool = False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.init_config(config, tune)

        sm_version = get_sm_version()
        if (
            sm_version == 90
            and num_tokens >= self.config["template_min_m"]
            and ffn_size >= hidden_size
        ):
            template_config = {
                key: self.config[key] for key in ("block_m", "block_n", "block_k", "num_stages")
            }
            template_config.update(num_math_wgs=2, swizzle_group_m=16)
            self._gemm_gate_up = GemmTemplate(
                GemmType.DENSE,
                static_dims="mnk",
                config=template_config,
            )
            self._gemm_down = GemmTemplate(
                GemmType.DENSE,
                static_dims="mnk",
                config=template_config,
            )
        elif sm_version == 90:
            gemm_config = self.config if config is not None else None
            self._gemm_gate_up = GemmKernel(
                m=num_tokens,
                n=ffn_size * 2,
                k=hidden_size,
                dtype=dtype,
                trans_b=True,
                config=gemm_config,
            )
            self._gemm_down = GemmKernel(
                m=num_tokens,
                n=hidden_size,
                k=ffn_size,
                dtype=dtype,
                trans_b=True,
                config=gemm_config,
            )
        else:
            self._gemm_gate_up = GemmBasicKernel(
                m=num_tokens,
                n=ffn_size * 2,
                k=hidden_size,
                dtype=dtype,
                trans_b=True,
                config=self.config,
            )
            self._gemm_down = GemmBasicKernel(
                m=num_tokens,
                n=hidden_size,
                k=ffn_size,
                dtype=dtype,
                trans_b=True,
                config=self.config,
            )

    @property
    def default_config(self) -> dict:
        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
            "threads": 256,
            "template_min_m": 512,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        return [self.default_config]

    def forward(
        self, hidden: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        T_dim = self.num_tokens
        F = self.ffn_size

        gate_up_out = self._gemm_gate_up(hidden, w_gate_up)

        silu_mul_fn = _silu_mul_fused_kernel(T_dim, F, self.dtype_str)(
            self.config["block_m"], self.config["block_n"], self.config["threads"]
        )
        gate_up = silu_mul_fn(gate_up_out)

        return self._gemm_down(gate_up, w_down)
