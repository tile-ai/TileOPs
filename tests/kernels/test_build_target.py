"""Target isolation for kernel caches and tuning."""

import dataclasses
import functools
import importlib
import inspect
import pkgutil
import types

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.jit import JITImpl

import tileops._kernels
from tests.build_target_facts import arch, compiles, described, other_arch, sm
from tileops._kernels.autotune import autotune
from tileops._kernels.build_target import BuildTarget, building_on, target_cache
from tileops.backend import DeviceFacts, _dispatch

# The code of every function ``target_cache`` returns.
_TARGET_KEYED = target_cache()(lambda: None).__code__


def _unkeyed(value: object, seen: "set[int] | None" = None) -> bool:
    """Find target-unaware caches reachable from imported objects.

    Objects created or rebound only at call time are outside this audit.
    """
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, JITImpl):
        return True
    if isinstance(value, functools._lru_cache_wrapper):
        return _unkeyed(value.__wrapped__, seen)
    if isinstance(value, types.FunctionType) and value.__code__ is _TARGET_KEYED:
        # Its own memo cache holds the target; a JIT object it wraps directly compiles through it.
        held = [] if isinstance(value.__wrapped__, JITImpl) else [value.__wrapped__]
    elif isinstance(value, (staticmethod, classmethod)):
        held = [value.__func__]
    elif isinstance(value, dict):
        held = list(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        held = list(value)
    elif isinstance(value, functools.partial):
        held = [value.func, *value.args, *value.keywords.values()]
    elif isinstance(value, types.FunctionType):
        held = [_contents(cell) for cell in value.__closure__ or ()]
        held += [value.__wrapped__] if hasattr(value, "__wrapped__") else []
    else:
        held = []
    return any(_unkeyed(item, seen) for item in held)


def _contents(cell: types.CellType) -> object:
    """What *cell* holds, ``None`` while it is unbound."""
    try:
        return cell.cell_contents
    except ValueError:
        return None


@pytest.mark.smoke
def test_no_kernel_cache_leaves_the_target_out():
    """No module or class in ``tileops._kernels`` holds a cache of builds without the target."""
    unkeyed = []
    for info in pkgutil.walk_packages(tileops._kernels.__path__, "tileops._kernels."):
        module = importlib.import_module(info.name)
        owners = [module, *(c for _, c in inspect.getmembers(module, inspect.isclass))]
        for owner in owners:
            unkeyed += [
                f"{getattr(owner, '__qualname__', module.__name__)}.{name}"
                for name, value in vars(owner).items()
                if _unkeyed(value)
            ]
    assert unkeyed == []


def _closing_over_one_jit():
    jit = tilelang.jit(out_idx=[-1])(_add_one)
    return target_cache()(lambda n: jit(n))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "make, unkeyed",
    [
        pytest.param(lambda: tilelang.jit(_add_one), True, id="bare"),
        pytest.param(lambda: {"f": tilelang.jit(_add_one)}, True, id="registry"),
        pytest.param(_closing_over_one_jit, True, id="closure"),
        pytest.param(lambda: functools.partial(tilelang.jit(_add_one), 256), True, id="partial"),
        pytest.param(lambda: target_cache()(tilelang.jit(_add_one)), False, id="wrapped"),
        pytest.param(lambda: functools.cache(lambda n: n + 1), False, id="pure-cache"),
        pytest.param(lambda: functools.cache(tilelang.jit(_add_one)), True, id="cached-jit"),
        pytest.param(
            lambda: functools.cache(functools.partial(tilelang.jit(_add_one), 256)),
            True,
            id="cached-partial-jit",
        ),
        pytest.param(
            lambda: target_cache()(lambda n: tilelang.jit(_add_one)(n)), False, id="per-build"
        ),
    ],
)
def test_the_cache_audit_finds_a_jit_object_however_it_is_held(make, unkeyed):
    """A JIT object counts as keyed only where ``target_cache`` wraps it directly or a build
    makes a new one."""
    assert _unkeyed(make()) is unkeyed


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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "fact, value",
    [("sm_count", 1), ("smem_budget", 1024), ("calibration", "other")],
)
def test_a_builder_builds_once_for_each_target(fact, value, monkeypatch):
    """A memoized builder builds again for a device that differs in any one fact, and each
    target reuses its own build."""
    read_facts = _dispatch.device_facts_and_calibration

    def changed(device):
        facts, calibration = read_facts(device)
        if fact == "calibration":
            calibration = value
        else:
            facts = dataclasses.replace(facts, **{fact: value})
        return facts, calibration

    builds = []
    builder = target_cache(maxsize=8)(lambda n: builds.append(n) or object())
    real = builder(1)
    with monkeypatch.context() as patch:
        patch.setattr(_dispatch, "device_facts_and_calibration", changed)
        other = builder(1)
        assert builder(1) is other
    assert other is not real and builder(1) is real and builds == [1, 1]


@pytest.mark.cuda_only("cuda-kernel")
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


@pytest.mark.cuda_only("cuda-kernel")
@pytest.mark.smoke
def test_tuning_compiles_its_candidates_for_the_target():
    """Worker compilation must retain the target without polluting the original JIT cache."""
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
