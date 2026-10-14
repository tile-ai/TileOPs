"""FP8 NT GEMM for per-tensor scales and at most eight rows, as the transposed product.

At ``m <= 8`` a 64-row WGMMA tile of ``a`` is mostly zero padding, and every K step
multiplies the padding by a full ``b`` tile. Computing ``c^T = b @ a^T`` instead puts 64
rows of ``b`` on WGMMA's 64-row axis and the rows of ``a``, padded to eight, on its
column axis: each K step is an ``m64n8`` product rather than an ``m64nN`` one, and the
grid is one CTA per 64 columns of the output.
"""

import functools
from typing import Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops.kernels.gemm.call_spec import GemmFP8Call, GemmFP8FwdInterface
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GemmFP8SwapABKernel"]


@functools.lru_cache(maxsize=32)
def _gemm_fp8_swap_ab_kernel(m: int, n: int, k: int, out_dtype: str) -> Callable:
    """Build the transposed per-tensor FP8 GEMM; the returned factory takes the ring depth.

    Args:
        m: Rows of ``a``, at most eight.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim, a multiple of 16 (TMA's 16-byte unit in fp8).
        out_dtype: ``"bfloat16"`` or ``"float16"``.

    Returns:
        A ``@tilelang.jit`` factory; calling it with ``num_stages`` returns the compiled
        ``prim_func``.
    """
    # WGMMA's row axis, which the 64 rows of ``b`` a CTA owns fill.
    block_nn = 64
    # The narrowest WGMMA column extent; the rows of ``a`` past ``m`` are TMA zero fill.
    m_pad = 8
    block_k = 128

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16", "-DENABLE_FP8"])
    def _gemm_fp8_swap_ab_func(num_stages: int = 4) -> Callable:
        @T.prim_func
        def _gemm_fp8_swap_ab_main(
            a: T.Tensor((m, k), "float8_e4m3fn"),  # type: ignore
            b: T.Tensor((n, k), "float8_e4m3fn"),  # type: ignore
            scale_a: T.Tensor((1, 1), "float32"),  # type: ignore
            scale_b: T.Tensor((1, 1), "float32"),  # type: ignore
            c: T.Tensor((m, n), out_dtype),  # type: ignore
        ) -> None:
            with T.Kernel(T.ceildiv(n, block_nn), threads=128) as bx:
                b_smem = T.alloc_shared((block_nn, block_k), "float8_e4m3fn")
                a_smem = T.alloc_shared((m_pad, block_k), "float8_e4m3fn")
                ct = T.alloc_fragment((block_nn, m_pad), "float32")
                ct_cast = T.alloc_fragment((block_nn, m_pad), out_dtype)
                ct_smem = T.alloc_shared((block_nn, m_pad), out_dtype)
                T.clear(ct)
                for ki in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    # Each CTA reads its rows of ``b`` once; ``a`` is what L2 keeps.
                    T.copy(b[bx * block_nn, ki * block_k], b_smem, eviction_policy="evict_first")
                    T.copy(a[0, ki * block_k], a_smem)
                    T.gemm(b_smem, a_smem, ct, transpose_B=True)
                for i, j in T.Parallel(block_nn, m_pad):
                    ct_cast[i, j] = ct[i, j] * scale_a[0, 0] * scale_b[0, 0]
                T.copy(ct_cast, ct_smem)
                for mi, j in T.Parallel(m_pad, block_nn):
                    if mi < m and bx * block_nn + j < n:
                        c[mi, bx * block_nn + j] = ct_smem[j, mi]

        return _gemm_fp8_swap_ab_main

    return _gemm_fp8_swap_ab_func


class GemmFP8SwapABKernel(Kernel, GemmFP8FwdInterface):
    """FP8 NT GEMM for per-tensor scales and ``m <= 8``, computed as ``(b @ a^T)^T``.

    Args:
        m: Rows of ``a``, at most eight.
        n: Rows of ``b``, columns of the output.
        k: Contraction dim.
        dtype: Operand dtype; ``torch.float8_e4m3fn``.
        out_dtype: ``torch.bfloat16`` or ``torch.float16``.
        config: Kernel config override; unset keys take their default.
        device_index: The device the kernel is built for.
    """

    supported_archs = [90]
    preferred_over = frozenset({"gemm_fp8_tensor_scale"})

    # The rows of ``a`` one transposed tile holds: WGMMA's narrowest column extent.
    _MAX_M = 8

    @classmethod
    def refusal(cls, call: GemmFP8Call) -> "str | None":
        if call.scale_a_shape != (1, 1) or call.scale_b_shape != (1, 1):
            return "requires one scale per tensor"
        if call.m > cls._MAX_M:
            return f"m={call.m} exceeds the {cls._MAX_M} rows one transposed tile holds"
        if call.dtype != torch.float8_e4m3fn:
            return f"requires float8_e4m3fn operands, got {call.dtype}"
        if call.out_dtype not in (torch.bfloat16, torch.float16):
            return f"writes bfloat16 or float16, not {call.out_dtype}"
        if call.has_bias:
            return "does not serve a bias"
        if call.k == 0 or call.k % 16:
            return f"k={call.k} is not a positive multiple of 16 (TMA's 16-byte unit in fp8)"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: GemmFP8Call) -> Entry:
        index = call.device.index if call.device is not None else None
        identity = (call.m, call.n, call.k, call.dtype, call.out_dtype, index)
        return identity, lambda: cls(
            call.m, call.n, call.k, call.dtype, call.out_dtype, device_index=index
        )

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.kernel = _gemm_fp8_swap_ab_kernel(m, n, k, self.dtype_to_str(out_dtype))
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # The deepest ring the shared memory holds, at most every K step: a CTA's loads
        # are all of its work, so as many as fit go out before the first product.
        k_steps = -(-self.k // 128)
        # A ring stage: 64 rows of ``b`` and the eight padded rows of ``a``, one 128-wide K
        # step of fp8 each.
        stage_bytes = (64 + 8) * 128
        # The 64x8 transposed output tile staged in shared memory beside the ring, at 2
        # bytes an element.
        staging_bytes = 64 * 8 * 2
        budget = BLOCK_SHARED_BYTES_OPT_IN[90] - staging_bytes
        return {"num_stages": max(1, min(k_steps, budget // stage_bytes))}

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError(f"{type(self).__name__} has no bias epilogue")
        return self.kernel(**self.config)(a, b, scale_a, scale_b)
