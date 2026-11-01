"""Every build is keyed by the target it is built for
(docs/design/ops-design.md § Kernel selection).

A second architecture is described through the device-facts reader: the facts of a device are
replaced by those of an architecture of another major version, whose binaries the device
refuses to load. A build for that device compiles for the described architecture, so the device refuses to
load it, where a reused build of the real architecture would run.
"""

import functools
import importlib
import inspect
import pkgutil

import pytest
import tilelang
import tilelang.language as T
import torch

import tileops._kernels
import tileops._utils.utils as device_utils
from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import CompiledProgram
from tileops.norm import RMSNormFwdOp


def _arch(index: int) -> int:
    """The architecture device *index* reports."""
    major, minor = torch.cuda.get_device_capability(index)
    return major * 10 + minor


def _other_arch(index: int) -> int:
    """An architecture whose binaries device *index* cannot load."""
    major = torch.cuda.get_device_capability(index)[0]
    return 90 if major == 8 else 89


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


def _sm(arch: int) -> str:
    """How TileLang names SM *arch* in a target."""
    return f"sm_{arch}a" if arch >= 90 else f"sm_{arch}"


@pytest.mark.smoke
def test_no_kernel_memo_cache_leaves_the_target_out():
    """Every memoized function in ``tileops._kernels`` is a ``target_cache``, whichever way it
    was made, so no cache of a build is keyed without the target."""
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
                if isinstance(value, functools._lru_cache_wrapper) and (
                    getattr(value, "__module__", None) == module.__name__
                ):
                    unkeyed.append(f"{module.__name__}.{getattr(owner, '__name__', '')}.{name}")
    assert unkeyed == []


@target_cache(maxsize=8)
@tilelang.jit(out_idx=[-1])
def _add_one(n: int):
    @T.prim_func
    def main(a: T.Tensor((n,), "float32"), b: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
            for i in T.Parallel(128):
                b[bx * 128 + i] = a[bx * 128 + i] + 1.0

    return main


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
    assert str(real.target.attrs["arch"]) != _sm(_other_arch(index))
    assert _add_one(256) is other
    describe_as_other_arch()
    assert _add_one(256) is real


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_an_op_compiles_for_the_target_of_its_device(describe_as_other_arch):
    """An op on a device described as another architecture compiles its own program for it,
    which the device refuses to load; the real target's op keeps computing the reference."""
    x = torch.randn(64, 1024, dtype=torch.float16, device="cuda")
    expected = torch.nn.functional.rms_norm(x.float(), (1024,), eps=1e-6).half()
    real = RMSNormFwdOp((1024,), eps=1e-6)
    torch.testing.assert_close(real(x), expected, rtol=1e-2, atol=1e-2)

    describe_as_other_arch(x.device.index)
    other = RMSNormFwdOp((1024,), eps=1e-6)
    with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
        other(x)
    describe_as_other_arch()
    torch.testing.assert_close(real(x), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second CUDA device")
def test_one_op_on_two_targets_builds_a_kernel_for_each(describe_as_other_arch):
    """One op called on devices of two targets holds a kernel for each; a program first
    launched while the other device is current compiles for its own device's target."""
    describe_as_other_arch(1)
    op = RMSNormFwdOp((1024,), eps=1e-6)
    first = torch.randn(64, 1024, dtype=torch.float16, device="cuda:0")
    expected = torch.nn.functional.rms_norm(first.float(), (1024,), eps=1e-6).half()
    with torch.accelerator.device_index(0):
        torch.testing.assert_close(op(first), expected, rtol=1e-2, atol=1e-2)
        with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
            op(first.to("cuda:1"))
        torch.testing.assert_close(op(first), expected, rtol=1e-2, atol=1e-2)
    archs = [_program_archs(kernel) for kernel in op.iter_kernels()]
    assert sorted(archs, key=sorted) == sorted([{_sm(_arch(0))}, {_sm(_other_arch(1))}], key=sorted)
