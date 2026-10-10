"""A kernel tuned after its first launch runs the program of its tuned config.

``Op.autotune`` reaches the kernels an op has already built. A kernel that compiled its
program from the config it held at that point compiles again from the config tuning
assigns. Each case calls an op once, has tuning assign a known config to every built
kernel, calls the op again, and checks the result and the program that ran.
"""

import types
from typing import Any

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.backend import BUILTIN
from tileops.kernels.kernel_base import Kernel
from tileops.ops import BmmFP8FwdOp, BmmFwdOp, GemmFwdOp
from tileops.ops.elementwise import ReluFwdOp, SoftplusFwdOp
from tileops.ops.op_base import Op
from workloads.elementwise import UnaryActivationCase
from workloads.gemm import BmmFP8Workload, BmmWorkload, GemmWorkload

pytestmark = [pytest.mark.cuda_only, pytest.mark.smoke]


class _ActivationTest(UnaryActivationCase, TestBase):
    pass


class _BmmTest(BmmWorkload, TestBase):
    pass


class _BmmFP8Test(BmmFP8Workload, TestBase):
    pass


class _GemmTest(GemmWorkload, TestBase):
    pass


class _LaunchRecord:
    """The builder arguments of each program a kernel launched, in launch order."""

    def __init__(self, kernel: Kernel, tuned: dict) -> None:
        self.kernel = kernel
        self.tuned = tuned
        self.launches: list[dict[str, Any]] = []


def _another_candidate(kernel: Kernel) -> dict:
    """A tuning candidate that differs from the kernel's current config."""
    for candidate in kernel.autotune_configs:
        if any(kernel.config.get(key) != value for key, value in candidate.items()):
            return dict(candidate)
    raise AssertionError(f"every {type(kernel).__name__} candidate equals its current config")


def _retune_built_kernels(op: Op, monkeypatch: pytest.MonkeyPatch) -> list[_LaunchRecord]:
    """Run ``op.autotune()`` with each built kernel's sweep answering another candidate.

    The candidate replaces the measured sweep, so the tuned config is known and differs
    from the one the kernel launched with. Each kernel's program builder is wrapped so
    that every program it builds from here on records its builder arguments on launch.
    """
    records = []
    for kernel in op.iter_kernels():
        if kernel.autotune_configs is None:
            continue
        builder = kernel.kernel
        parameters = list(builder.signature.parameters)
        record = _LaunchRecord(kernel, _another_candidate(kernel))

        def build(*args, _builder=builder, _parameters=parameters, _record=record, **kwargs):
            program = _builder(*args, **kwargs)
            # Name each argument by the config key it carries (``npt_arg`` -> ``num_per_thread``).
            named = {**dict(zip(_parameters[: len(args)], args, strict=True)), **kwargs}
            aliases = Kernel._AUTOTUNE_PARAM_ALIASES
            arguments = {aliases.get(name, name): value for name, value in named.items()}

            def launch(*tensors, **options):
                _record.launches.append(arguments)
                return program(*tensors, **options)

            return launch

        def sweep(*args, _tuned=record.tuned, **kwargs):
            return types.SimpleNamespace(config=dict(_tuned))

        monkeypatch.setattr(kernel, "kernel", build)
        monkeypatch.setattr(kernel, "tune_jit_kernel", sweep)
        records.append(record)
    assert records, f"{type(op).__name__} built no tunable kernel"
    op.autotune()
    return records


def _check_tuned_launch(test: TestBase, op: Op, monkeypatch: pytest.MonkeyPatch) -> None:
    """Call *op*, tune what it built, call it again, and check what that call launched."""
    inputs = test.gen_inputs()
    op(*inputs)
    records = _retune_built_kernels(op, monkeypatch)
    test.check(op, *inputs)
    for record in records:
        kernel = record.kernel
        name = type(kernel).__name__
        assert record.tuned.items() <= kernel.config.items(), f"{name} kept {kernel.config}"
        assert record.launches, f"{name} launched the program compiled before tuning"
        launched = record.launches[-1]
        assert record.tuned.items() <= launched.items(), (
            f"{name} launched {launched}, tuned to {record.tuned}"
        )


@pytest.mark.parametrize("op_cls", [ReluFwdOp, SoftplusFwdOp])
def test_elementwise_launches_the_tuned_program(op_cls, monkeypatch) -> None:
    """``ReluFwdOp`` runs a unary strategy kernel and ``SoftplusFwdOp`` a multi-input one;
    both compile their program at construction."""
    test = _ActivationTest(4096, torch.float16, op_cls.__name__)
    _check_tuned_launch(test, op_cls(target=BUILTIN), monkeypatch)


def test_gemm_launches_the_tuned_program(monkeypatch) -> None:
    """K = 2 is TMA-misaligned, so the pipelined kernel serves it on every architecture;
    it compiles its program on the first launch."""
    test = _GemmTest(32, 64, 2, torch.bfloat16, False, True)
    _check_tuned_launch(test, GemmFwdOp(trans_b=True, target=BUILTIN), monkeypatch)


def test_bmm_launches_the_tuned_program(monkeypatch) -> None:
    """The batched kernel compiles its program on the first launch."""
    test = _BmmTest(4, 128, 128, 128, torch.float16)
    _check_tuned_launch(test, BmmFwdOp(target=BUILTIN), monkeypatch)


@pytest.mark.parametrize(
    "arch",
    [
        # The FP8 programs run on SM89 and SM90 only. The warp-specialized and persistent
        # ones are SM90-only, so SM89 runs the classic 3D-grid program instead.
        pytest.param("sm90", marks=pytest.mark.sm90, id="sm90"),
        pytest.param("sm89", marks=pytest.mark.sm89, id="sm89"),
    ],
)
def test_bmm_fp8_launches_the_tuned_programs(arch, monkeypatch) -> None:
    """A ``[B, K, N]`` ``b`` also builds the transpose kernel; both compile their program
    on the first launch."""
    test = _BmmFP8Test(4, 128, 128, 128, torch.float8_e4m3fn, out_dtype=torch.bfloat16)
    _check_tuned_launch(test, BmmFP8FwdOp(out_dtype=torch.bfloat16, target=BUILTIN), monkeypatch)
