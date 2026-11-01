"""Every build is keyed by the target it is built for
(docs/design/ops-design.md § Kernel selection).

No host here has devices of two architectures, so a second one is described through the
device-facts reader: a device's facts are replaced by those of an architecture of another
major version. A build for that device compiles for the described architecture, which the
device refuses to load; a reused build of the real architecture would run.
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
import tileops._utils.utils as device_utils
from tests.workload_test_base import TestBase
from tileops._kernels.autotune import autotune
from tileops._kernels.build_target import building_on, target_cache
from tileops._kernels.kernel_base import CompiledProgram
from tileops.norm import RMSNormFwdOp
from workloads.norm import RMSNormWorkload


def _arch(index: int) -> int:
    """The architecture device *index* reports."""
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def _other_arch(index: int) -> int:
    """An architecture whose binaries device *index* cannot load."""
    return 90 if _arch(index) // 10 == 8 else 89


def _sm(arch: int) -> str:
    """How TileLang names SM *arch* in a target."""
    return f"sm_{arch}a" if arch >= 90 else f"sm_{arch}"


@pytest.fixture
def describe_as_other_arch():
    """Describe the devices of the indices passed as another architecture until the test ends."""
    real = device_utils._device_facts

    def describe(*indices: int) -> None:
        def facts(index: int):
            arch, calibration, sm_count, smem_budget = real(index)
            if index in indices:
                return _other_arch(index), None, sm_count, smem_budget
            return arch, calibration, sm_count, smem_budget

        device_utils.forget_device_properties()
        device_utils._device_facts = functools.lru_cache(maxsize=16)(facts)

    yield describe
    device_utils._device_facts = real
    device_utils.forget_device_properties()


def _program_archs(kernel) -> set[str]:
    """The CUDA architectures the programs *kernel* holds were compiled for."""
    archs = set()
    for program in kernel._generation.programs.values():
        for one in program if isinstance(program, tuple) else (program,):
            jit = one.jit if isinstance(one, CompiledProgram) else one
            archs.add(str(jit.target.attrs["arch"]))
    return archs


@pytest.mark.smoke
def test_no_kernel_cache_leaves_the_target_out():
    """No module or class in ``tileops._kernels`` holds a ``functools`` memo cache or a bare JIT
    object, whichever way it was made: both cache builds without the target."""
    unkeyed = []
    for info in pkgutil.walk_packages(tileops._kernels.__path__, "tileops._kernels."):
        module = importlib.import_module(info.name)
        if info.name == "tileops._kernels.build_target":
            continue
        owners = [module, *(c for _, c in inspect.getmembers(module, inspect.isclass))]
        for owner in owners:
            for name, value in vars(owner).items():
                if isinstance(value, (staticmethod, classmethod)):
                    value = value.__func__
                if isinstance(value, JITImpl) or (
                    isinstance(value, functools._lru_cache_wrapper)
                    and value.__module__ == module.__name__
                ):
                    unkeyed.append(f"{getattr(owner, '__qualname__', module.__name__)}.{name}")
    assert unkeyed == []


def _add_one_program(n: int):
    @T.prim_func
    def main(a: T.Tensor((n,), "float32"), b: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for i in T.Parallel(128):
                b[bx * 128 + i] = a[bx * 128 + i] + 1.0

    return main


_add_one = target_cache(maxsize=8)(tilelang.jit(out_idx=[-1])(_add_one_program))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_a_jit_builder_compiles_once_for_each_target(describe_as_other_arch):
    """A memoized JIT builder compiles a program for each target it is called for, and reuses
    each target's own."""
    index = torch.accelerator.current_device_index()
    real = _add_one(256)
    describe_as_other_arch(index)
    other = _add_one(256)
    assert str(other.target.attrs["arch"]) == _sm(_other_arch(index))
    assert str(real.target.attrs["arch"]) == _sm(_arch(index))
    assert _add_one(256) is other
    describe_as_other_arch()
    assert _add_one(256) is real


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_tuning_compiles_its_candidates_for_the_target(describe_as_other_arch):
    """TileLang compiles tuning candidates on worker threads, which hold no target scope; the
    JIT object it tunes states the caller's target, and is not the builder's own."""
    index = torch.accelerator.current_device_index()
    jit = tilelang.jit(out_idx=[-1])(_add_one_program)
    describe_as_other_arch(index)
    with building_on(None):
        tuner = autotune(configs=[{"n": 256}])(jit)
    assert str(tuner.jit_impl.target.attrs["arch"]) == _sm(_other_arch(index))
    assert tuner.jit_impl is not jit


@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_an_op_compiles_for_the_target_of_its_device(describe_as_other_arch):
    """An op on a device described as another architecture compiles its own program for it,
    which the device refuses to load; an op on the real target keeps its own program."""
    workload = RMSNormWorkload(64, 1024, torch.float16)
    inputs = workload.gen_inputs()
    real = RMSNormFwdOp((1024,), eps=workload.eps)
    TestBase.check(workload, real, *inputs)

    describe_as_other_arch(inputs[0].device.index)
    other = RMSNormFwdOp((1024,), eps=workload.eps)
    with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
        other(*inputs)
    describe_as_other_arch()
    TestBase.check(workload, real, *inputs)


@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second CUDA device")
def test_one_op_on_two_targets_builds_a_kernel_for_each(describe_as_other_arch):
    """One op called on devices of two targets holds a kernel for each; a program first
    launched while the other device is current compiles for its own device's target."""
    describe_as_other_arch(1)
    workload = RMSNormWorkload(64, 1024, torch.float16)
    first = [t.to("cuda:0") for t in workload.gen_inputs()]
    op = RMSNormFwdOp((1024,), eps=workload.eps)
    with torch.accelerator.device_index(0):
        TestBase.check(workload, op, *first)
        with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
            op(*(t.to("cuda:1") for t in first))
        TestBase.check(workload, op, *first)
    archs = sorted(sorted(_program_archs(kernel)) for kernel in op.iter_kernels())
    assert archs == sorted([[_sm(_arch(0))], [_sm(_other_arch(1))]])
