"""Substitute device facts and observe compilation targets.

The substituted architecture cannot load on the real device, exposing accidental reuse.
"""

import contextlib
import functools
from collections.abc import Iterator

import torch
from tilelang.contrib.nvcc import get_target_arch
from tilelang.jit import JITImpl

import tileops._utils.utils as device_utils


def arch(index: int) -> int:
    """The architecture device *index* reports."""
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def other_arch(index: int) -> int:
    """An architecture whose binaries device *index* cannot load."""
    return 90 if arch(index) // 10 == 8 else 89


def sm(arch: int) -> str:
    """How TileLang names SM *arch* in a target, spelled as ``building_on`` spells it."""
    return f"sm_{get_target_arch((arch // 10, arch % 10))}"


@contextlib.contextmanager
def described(stated: "dict[int, dict]") -> Iterator[None]:
    """Temporarily override device facts by CUDA device index."""
    real = device_utils._device_facts

    def facts(index: int) -> device_utils.CudaFacts:
        return real(index)._replace(**stated.get(index, {}))

    device_utils.forget_device_properties()
    device_utils._device_facts = functools.lru_cache(maxsize=16)(facts)
    try:
        yield
    finally:
        device_utils._device_facts = real
        # After the reader is back, so no fact read under the description stays cached.
        device_utils.forget_device_properties()


@contextlib.contextmanager
def compiles() -> Iterator[list]:
    """The architecture of each program TileLang compiles inside the block, in order."""
    built = []
    compile_ = JITImpl.compile

    def recorded(self, *args, **kwargs):
        program = compile_(self, *args, **kwargs)
        built.append(str(program.target.attrs["arch"]))
        return program

    JITImpl.compile = recorded
    try:
        yield built
    finally:
        JITImpl.compile = compile_
