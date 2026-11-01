"""The target a build compiles for, and the cache every kernel builder memoizes through.

A build target is the device type and the device facts of the device a build runs for.
Every compiled program, and every value a builder derives from device facts, is keyed by it,
so a device of another architecture, SM count or shared-memory budget never reuses it.
"""

import contextlib
import functools
from typing import Any, Callable, Iterator, NamedTuple, Optional

import torch

from tileops._utils.device import current_device, device_context, on_current_index

__all__ = ["BuildTarget", "building_on", "current_target", "target_cache", "target_of"]


class BuildTarget(NamedTuple):
    """The device facts a build compiles for, with their device type."""

    kind: Optional[str]
    arch: int
    sm_count: int
    smem_budget: int
    calibration: Optional[str]


def target_of(device: Optional[torch.device]) -> BuildTarget:
    """The build target of *device*; ``None`` names the current device.

    Read through the one device-facts reader, so a backend's device type reports its own.
    """
    from tileops.backend._dispatch import device_facts_and_calibration

    device = current_device() if device is None else on_current_index(device)
    facts, calibration = device_facts_and_calibration(device)
    kind = None if device is None else device.type
    return BuildTarget(kind, facts.arch, facts.sm_count, facts.smem_budget, calibration)


def current_target() -> BuildTarget:
    """The build target of the current device: what a build running now compiles for."""
    return target_of(None)


@contextlib.contextmanager
def building_on(device: Optional[torch.device]) -> Iterator[None]:
    """Build for *device* inside the block: make it current and compile for its target.

    TileLang compiles ``target="auto"`` for the innermost target scope, so the scope states
    the architecture TileOPs read for *device* rather than leaving TileLang to detect one.
    It spells the architecture as TileLang's detection does, so a program compiled under it
    shares TileLang's cache entries with a detected one.
    """
    target = target_of(device)
    with device_context(device):
        if target.kind != "cuda" or target.arch <= 0:
            yield
            return
        from tilelang.contrib.nvcc import get_target_arch
        from tvm.target import Target

        sm = get_target_arch((target.arch // 10, target.arch % 10))
        with Target({"kind": "cuda", "arch": f"sm_{sm}"}):
            yield


def target_cache(maxsize: Optional[int] = 128) -> Callable[[Callable], Callable]:
    """``functools.lru_cache`` whose key also holds :func:`current_target`.

    Every memoized function in ``tileops._kernels`` goes through this: a builder's result is a
    JIT object, a compiled program or a value read from device facts, each of which belongs to
    the target it was built for. Decorating a ``@tilelang.jit`` object compiles each miss
    through ``compile``, which skips the JIT object's own cache: that one is keyed by the
    arguments alone.
    """

    def decorate(fn: Callable) -> Callable:
        from tilelang.jit import JITImpl

        jit = isinstance(fn, JITImpl)

        @functools.lru_cache(maxsize=maxsize)
        def cached(target: BuildTarget, *args: Any, **kwargs: Any) -> Any:
            # Compiles for the target it is keyed by, as TileLang would detect it.
            with building_on(None):
                return fn.compile(*args, **kwargs) if jit else fn(*args, **kwargs)

        @functools.wraps(fn, updated=())
        def keyed(*args: Any, **kwargs: Any) -> Any:
            return cached(current_target(), *args, **kwargs)

        keyed.cache_clear = cached.cache_clear
        keyed.cache_info = cached.cache_info
        return keyed

    return decorate
