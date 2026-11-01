"""Describe a device as another build target, and observe what TileLang compiles.

A build-target isolation test substitutes device facts through the one device-facts reader
and runs the normal dispatch (docs/design/testing.md § Tests). The architecture it describes
is of another major version than the device's, so the device refuses to load a program built
for it, where a reused program of the real architecture would run.
"""

import contextlib
import functools
from collections.abc import Iterator

import torch
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
    """How TileLang names SM *arch* in a target."""
    return f"sm_{arch}a" if arch >= 90 else f"sm_{arch}"


@contextlib.contextmanager
def described(stated: "dict[int, dict]") -> Iterator[None]:
    """Inside the block, device ``cuda:<i>`` reports the facts ``stated[i]`` names in place of
    its own: ``arch``, ``calibration``, ``sm_count`` or ``smem_budget``."""
    real = device_utils._device_facts
    names = ("arch", "calibration", "sm_count", "smem_budget")

    def facts(index: int) -> tuple:
        own = dict(zip(names, real(index), strict=True))
        return tuple({**own, **stated.get(index, {})}[name] for name in names)

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
