"""Every build is keyed by the target it is built for
(docs/design/ops-design.md § Kernel selection): the memo cache builders go through, the tuning
wrapper, and the absence of any other cache of a build in ``tileops._kernels``.
"""

import functools
import importlib
import inspect
import pkgutil

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.jit import JITImpl

import tileops._kernels
from tests.build_target_facts import arch, compiles, described, other_arch, sm
from tileops._kernels.autotune import autotune
from tileops._kernels.build_target import BuildTarget, building_on, target_cache
from tileops.backend import DeviceFacts


def _held(value: object) -> list:
    """*value*, the function a static or class method wraps, and what a container holds."""
    if isinstance(value, (staticmethod, classmethod)):
        return [value.__func__]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [held for item in value for held in _held(item)]
    return [value]


@pytest.mark.smoke
def test_no_kernel_cache_leaves_the_target_out():
    """No module or class in ``tileops._kernels`` holds a ``functools`` memo cache or a bare JIT
    object, directly or in a container: both cache builds without the target."""
    unkeyed = []
    for info in pkgutil.walk_packages(tileops._kernels.__path__, "tileops._kernels."):
        module = importlib.import_module(info.name)
        if info.name == "tileops._kernels.build_target":
            continue
        owners = [module, *(c for _, c in inspect.getmembers(module, inspect.isclass))]
        for owner in owners:
            for name, value in vars(owner).items():
                for held in _held(value):
                    if isinstance(held, JITImpl) or (
                        isinstance(held, functools._lru_cache_wrapper)
                        and held.__module__ == module.__name__
                    ):
                        unkeyed.append(f"{getattr(owner, '__qualname__', module.__name__)}.{name}")
    assert unkeyed == []


@pytest.mark.smoke
def test_the_build_target_holds_every_device_fact():
    """A fact added to the device facts is a fact a build may read, so it keys builds too."""
    facts = {f.name for f in DeviceFacts.__dataclass_fields__.values()}
    assert set(BuildTarget._fields) == {"kind", "calibration", *facts}


def _add_one(n: int, threads: int = 128):
    @T.prim_func
    def main(a: T.Tensor((n,), "float32"), b: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                b[bx * threads + i] = a[bx * threads + i] + 1.0

    return main


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "fact, value",
    [("arch", None), ("sm_count", 1), ("smem_budget", 1024), ("calibration", "other")],
)
def test_a_builder_builds_once_for_each_target(fact, value):
    """A memoized builder builds again for a device that differs in any one fact, and each
    target reuses its own build."""
    index = torch.accelerator.current_device_index()
    builds = []
    builder = target_cache(maxsize=8)(lambda n: builds.append(n) or object())
    real = builder(1)
    with described({index: {fact: other_arch(index) if value is None else value}}):
        other = builder(1)
        assert builder(1) is other
    assert other is not real and builder(1) is real and builds == [1, 1]


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_a_jit_builder_compiles_for_each_target():
    """A memoized JIT object compiles a program for each target it is called for, though its
    own cache is keyed by the arguments alone, and reuses each target's program."""
    index = torch.accelerator.current_device_index()
    builder = target_cache(maxsize=8)(tilelang.jit(out_idx=[-1])(_add_one))
    with compiles() as built:
        real = builder(256)
        with described({index: {"arch": other_arch(index)}}):
            other = builder(256)
            assert builder(256) is other
        assert builder(256) is real
    assert built == [sm(arch(index)), sm(other_arch(index))]


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_tuning_compiles_its_candidates_for_the_target():
    """TileLang compiles tuning candidates on worker threads, which hold no target scope; each
    candidate still compiles for the caller's target, which the device refuses to load, and
    the builder's own JIT object keeps none of them."""
    index = torch.accelerator.current_device_index()
    jit = tilelang.jit(out_idx=[-1])(_add_one)
    tune = autotune(configs=[{"threads": 64}, {"threads": 128}], warmup=1, rep=1)
    with compiles() as built:
        with (
            described({index: {"arch": other_arch(index)}}),
            building_on(None),
            pytest.raises(RuntimeError, match="No configuration successfully"),
        ):
            tune(jit)(256)
        assert built == [sm(other_arch(index))] * 2
        jit(256, threads=64)
    assert built[-1] == sm(arch(index))
