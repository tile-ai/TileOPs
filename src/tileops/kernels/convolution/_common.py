"""What the 1d, 2d and 3d kernels share: the autotune search space and the launch path."""

import itertools
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel


def conv_autotune_configs(
    dtype,
    *,
    block_m=(32, 64, 128),
    block_n=(64, 128, 256),
    block_k=(32, 64, 128),
    num_stages=(2, 3),
    threads=(128, 256),
) -> list[dict]:
    """Search space filtered to combinations that fit in shared memory."""
    limit = get_shared_memory_limit_bytes()
    valid = []
    for bm, bn, bk, ns, th in itertools.product(
        block_m,
        block_n,
        block_k,
        num_stages,
        threads,
    ):
        if conv_shared_memory_bytes(bm, bn, bk, ns, dtype) > limit:
            continue
        valid.append(
            {
                "block_m": bm,
                "block_n": bn,
                "block_k": bk,
                "num_stages": ns,
                "threads": th,
                "enable_rasterization": True,
            }
        )
    return valid


def get_shared_memory_limit_bytes() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin


def conv_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    out_shared_bytes = block_m * block_n * dtype_bytes
    return per_stage_bytes * max(1, num_stages) + out_shared_bytes


def _launch(
    kernel: Kernel,
    *tensors: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run *kernel*'s compiled program, passing *bias* only when the call carries one.

    The program was traced for one side of ``has_bias``: the no-bias variant has no bias
    parameter at all, so what the call carries has to agree with what was built.

    Args:
        kernel: The kernel whose ``self.kernel`` builder and ``self.config`` are used.
        *tensors: The program's inputs in prim_func order, bias excluded.
        bias: The bias this call carries, or ``None``.

    Returns:
        The output tensor the program allocated.

    Raises:
        ValueError: The call's bias presence differs from the one the kernel was built for.
    """
    if (bias is not None) != kernel.has_bias:
        built, given = ("with", "without") if kernel.has_bias else ("without", "with")
        raise ValueError(
            f"{type(kernel).__name__} was built {built} a bias and was called {given} one; "
            f"bias presence is part of what the program is compiled for, so the op layer "
            f"builds one kernel per side"
        )
    config = kernel.config
    program = kernel.kernel(
        config["block_m"],
        config["block_n"],
        config["block_k"],
        config["num_stages"],
        config["threads"],
        config["enable_rasterization"],
    )
    if bias is None:
        return program(*tensors)
    return program(*tensors, bias)
