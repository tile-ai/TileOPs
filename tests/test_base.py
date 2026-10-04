from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from typing import Any

import pytest
import torch

from tileops.backend import BUILTIN
from workloads.numerics import verify
from workloads.workload_base import FixtureBase, FixtureMeta, WorkloadBase

_logger = logging.getLogger("tileops.ops")

# Thread-local storage for conftest hook to pick up per-test Op info.
_check_result = threading.local()


def _refuse_non_op(op: object, op_name: str) -> None:
    """Raise unless *op* is an Op, which is what the result is reported under.

    Raised before the reference runs, so a test handing check() a kernel or a
    compiled callable is told where that belongs -- in ``runs=`` -- rather than
    producing a result filed under a name that is not an op.
    """
    from tileops.ops.op_base import Op

    if not isinstance(op, Op):
        raise AssertionError(
            f"check() takes the Op the result belongs to, got {op_name}; "
            f"pass what to execute as runs="
        )


# Canonical import hub: tests import fixture types from here, not workloads.
__all__ = [
    "FixtureBase",
    "FixtureMeta",
    "TestBase",
    "served_in_tree",
]


def served_in_tree(op: Any) -> bool:
    """Whether a call settled *op* on the in-tree implementation.

    Gates an assertion about the in-tree kernels, so the rest of the test also runs
    against a backend that serves the op.
    """
    return op.settled_target is BUILTIN


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
        runs supplies a kernel or compiled implementation owned by the Op.
        """
        _check_result.op_name = None
        _check_result.op_module = None
        _check_result.max_abs_err = None
        _check_result.checked_outputs = 0
        name, module = type(op).__name__, type(op).__module__
        _refuse_non_op(op, name)
        _check_result.op_name = name
        _check_result.op_module = module
        result = verify(
            op if runs is None else runs,
            inputs,
            reference=self.ref_program,
            evidence=self.verification(*inputs),
        )
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
