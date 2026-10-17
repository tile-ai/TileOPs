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
from tileops.kernels.gemm.dense import swap_ab_mainloop, swap_ab_store
from tileops.kernels.gemm.heuristics import SWAP_AB_MPAD
from tileops.kernels.kernel_base import Entry, Kernel

__all__ = ["GemmFP8SwapABKernel"]

# The transposed tile, which the builder allocates, admission bounds and the stage budget
# sizes: 64 rows of ``b`` on WGMMA's row axis, the rows of ``a`` padded to WGMMA's narrowest
# column extent, 128-wide K steps.
_BLOCK_NN = 64
_BLOCK_K = 128


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
    block_nn, block_k = _BLOCK_NN, _BLOCK_K
    # The rows of ``a`` past ``m`` are TMA zero fill.
    m_pad = SWAP_AB_MPAD

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
                swap_ab_mainloop(a, b, ct, a_smem, b_smem, bx, k, block_nn, block_k, num_stages)
                for i, j in T.Parallel(block_nn, m_pad):
                    ct_cast[i, j] = ct[i, j] * scale_a[0, 0] * scale_b[0, 0]
                swap_ab_store(ct_cast, ct_smem, c, bx, m, n, block_nn, m_pad)

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

    @classmethod
    def refusal(cls, call: GemmFP8Call) -> "str | None":
        if call.scale_a_shape != (1, 1) or call.scale_b_shape != (1, 1):
            return "requires one scale per tensor"
        if call.m > SWAP_AB_MPAD:
            return f"m={call.m} exceeds the {SWAP_AB_MPAD} rows one transposed tile holds"
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
        k_steps = -(-self.k // _BLOCK_K)
        # A ring stage: the tile's rows of ``b`` and the padded rows of ``a``, one K step
        # of fp8 each.
        stage_bytes = (_BLOCK_NN + SWAP_AB_MPAD) * _BLOCK_K
        # The transposed output tile staged in shared memory beside the ring, at 2 bytes
        # an element.
        staging_bytes = _BLOCK_NN * SWAP_AB_MPAD * 2
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
