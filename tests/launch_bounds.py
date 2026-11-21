"""The ``__launch_bounds__`` an elementwise op's built program declares."""

import re

import torch

from workloads.device import run_device


def elementwise_launch_bounds(op) -> tuple[int, list[tuple[str, str]]]:
    """The threads of the one program *op* has built and its ``__launch_bounds__`` pairs."""
    (kernel,) = op._entries_built("elementwise").values()
    source = kernel.program(kernel._compile).get_kernel_source()
    return kernel.config["threads"], re.findall(r"__launch_bounds__\((\d+), (\d+)\)", source)


def full_occupancy_bounds(threads: int) -> list[tuple[str, str]]:
    """The ``__launch_bounds__`` of *threads*-thread blocks enough to fill an SM's register file
    at 32 registers a thread."""
    resident = torch.cuda.get_device_properties(run_device()).regs_per_multiprocessor // 32
    return [(str(threads), str(resident // threads))]
