"""What the 1d, 2d and 3d kernels share: the swizzle panel, the output tile's epilogue, the
autotune search space, the grid limit and the launch path."""

import itertools

import tilelang.language as T
import torch

from tileops._kernels.constants import MAX_GRID_YZ_BLOCKS
from tileops._kernels.kernel_base import Kernel

# The panel every convolution kernel's block swizzle groups blocks into: a power of two so the
# block remap divides by shift, and the narrowest one because long grids lose as it widens.
SWIZZLE_PANEL = 2


@T.macro
def stage_output_tile(
    out_local, out_shared, bias, block_m, block_n, row0, rows, col0, cols, channel0, dtype
):
    """Cast the accumulator tile into ``out_shared``, adding the bias where there is one.

    Row ``i`` of the tile is output channel ``channel0 + row0 + i`` and column ``j`` is
    output position ``col0 + j``. An element past ``rows`` or ``cols`` is stored as zero.
    """
    for i, j in T.Parallel(block_m, block_n):
        row = row0 + i
        col = col0 + j
        if bias is not None:
            out_shared[i, j] = T.if_then_else(
                (row < rows) & (col < cols),
                T.cast(out_local[i, j] + T.cast(bias[channel0 + row], out_local.dtype), dtype),
                T.cast(0.0, dtype),
            )
        else:
            out_shared[i, j] = T.if_then_else(
                (row < rows) & (col < cols),
                T.cast(out_local[i, j], dtype),
                T.cast(0.0, dtype),
            )


def grid_refusal(*, y: int = 1, z: int = 1) -> str | None:
    """Why a launch of ``y`` blocks along grid y and ``z`` along z cannot run, or ``None``."""
    for axis, blocks in (("y", y), ("z", z)):
        if blocks > MAX_GRID_YZ_BLOCKS:
            return f"launches {blocks} blocks along grid {axis}, past {MAX_GRID_YZ_BLOCKS}"
    return None


def operand_refusal(elements: int) -> str | None:
    """Why a program with an operand of ``elements`` elements cannot build, or ``None``.

    TileLang fails to build the programs that ask once an index into that operand passes
    int32. Each caller passes the largest of the operands its program fails on.
    """
    if elements > 2**31 - 1:
        return f"builds only for operands of at most 2**31 - 1 elements, and one holds {elements}"
    return None


def conv_num_stages(arch: int) -> int:
    """Pipeline depth the shared memory of architecture *arch* holds: three on SM90, two
    before."""
    return 3 if arch == 90 else 2


def conv_autotune_configs(
    dtype: torch.dtype,
    smem_budget: int,
    *,
    block_m=(32, 64, 128),
    block_n=(64, 128, 256),
    block_k=(32, 64, 128),
    threads=(128, 256),
    out_pad_cols: int = 0,
) -> list[dict]:
    """Search space filtered to combinations that fit in *smem_budget* bytes of shared memory.

    ``enable_rasterization`` turns on a swizzle that orders blocks for L2 locality. It is
    searched rather than fixed because which way wins follows the grid a shape produces.
    ``out_pad_cols`` is the number of columns a kernel pads its shared output tile with.
    """
    configs = []
    for bm, bn, bk, ns, th, rast in itertools.product(
        block_m, block_n, block_k, (2, 3), threads, (False, True)
    ):
        stage_bytes = (bm * bk + bk * bn) * dtype.itemsize
        out_bytes = bm * (bn + out_pad_cols) * dtype.itemsize
        if stage_bytes * ns + out_bytes > smem_budget:
            continue
        configs.append(
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "num_stages": ns,
                "threads": th,
                "enable_rasterization": rast,
            }
        )
    return configs


def launch(
    kernel: Kernel,
    *tensors: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Run *kernel*'s compiled program, passing *bias* only when the call carries one.

    The program was traced for one side of ``has_bias``: the no-bias variant has no bias
    parameter at all, so what the call carries has to agree with what was built.

    Args:
        kernel: The kernel whose published program runs; it is compiled from the
            ``self.kernel`` builder and the generation's config on its first launch.
        *tensors: The program's inputs in prim_func order, bias excluded.
        bias: The bias this call carries, or ``None``.

    Returns:
        The output tensor the program allocated.
    """
    # A config's keys are the builder's parameter names, as autotune binds them.
    program = kernel.program(lambda config: kernel.kernel(**config))
    if bias is None:
        return program(*tensors)
    return program(*tensors, bias)
