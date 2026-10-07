from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from typing import Any

import pytest
import torch

from workloads.numerics import Request, verify
from workloads.workload_base import FixtureBase, FixtureMeta, WorkloadBase

_logger = logging.getLogger("tileops.ops")

# Thread-local storage for conftest hook to pick up per-test Op info.
_check_result = threading.local()


def _refuse_non_op(op: object, op_name: str, runs: object) -> None:
    """Raise unless *op* is an Op and *runs* is not a kernel.

    The result is reported under *op*, and what executes reaches the kernel the way a
    caller does: through the op's dispatch. ``runs`` takes a compiled or wrapped form of
    the op; a kernel is pinned through ``kernel_map`` instead. Raised before the reference
    runs.
    """
    from tileops.kernels.kernel_base import Kernel
    from tileops.ops.op_base import Op

    if not isinstance(op, Op):
        raise AssertionError(
            f"check() takes the Op the result belongs to, got {op_name}; "
            f"pass a compiled or wrapped form of it as runs="
        )
    if isinstance(getattr(runs, "__self__", runs), Kernel):
        raise AssertionError(
            "runs= takes a compiled or wrapped form of the op; reach a kernel through the "
            "op's dispatch, pinning it with kernel_map"
        )


# Canonical import hub: tests import fixture types from here, not workloads.
__all__ = [
    "FixtureBase",
    "FixtureMeta",
    "TestBase",
]


class TestBase(WorkloadBase):
    """Abstract base class for op correctness testing.

    The concrete workload supplies gen_inputs(), ref_program() and verification().
    Provides check() for comparing op output against reference.
    """

    __test__ = False

    @abstractmethod
    def ref_program(self, *inputs: Any) -> Any:
        """Reference implementation for correctness checking.

        Supplied by the concrete workload. Returns the same outputs as the op under test,
        using a simpler/trusted implementation (e.g. PyTorch built-ins).
        """
        raise NotImplementedError

    def check(self, op, *inputs: torch.Tensor, runs=None) -> None:
        """Verify a workload's declaration and attribute the result to its Op.

        Numerical policy belongs to verification(), never to the call site.
        runs supplies a compiled or wrapped form of the op.
        """
        _check_result.op_name = None
        _check_result.op_module = None
        _check_result.max_abs_err = None
        _check_result.checked_outputs = 0
        name, module = type(op).__name__, type(op).__module__
        _refuse_non_op(op, name, runs)
        _check_result.op_name = name
        _check_result.op_module = module
        result = verify(
            self.ref_program,
            inputs,
            evidence=self.verification(*inputs),
            requests={name: Request(op if runs is None else runs, inputs)},
        )[name]
        if not result.checked_outputs:
            pytest.skip(result.unchecked_reason)
        _check_result.checked_outputs = result.checked_outputs
        _check_result.max_abs_err = result.max_abs_err
        _logger.info(
            "op=%s module=%s status=pass checked_outputs=%s max_abs_err=%s",
            name,
            module,
            result.checked_outputs,
            result.max_abs_err,
        )
