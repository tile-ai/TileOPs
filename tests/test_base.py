import logging
import threading
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Tuple

import pytest
import torch

_logger = logging.getLogger("tileops.ops")

# Thread-local storage for conftest hook to pick up per-test Op info.
_check_result = threading.local()


def _to_tuple(outputs):
    """Normalize outputs to a tuple."""
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, list):
        return tuple(outputs)
    if isinstance(outputs, tuple):
        return outputs
    raise ValueError(f"Unsupported output type: {type(outputs)}")


def allclose_compare(output: torch.Tensor, output_ref: torch.Tensor, atol: float = 1e-8, rtol: float = 1e-5) -> None:
    """Default comparison using torch.allclose."""
    output, output_ref = torch.broadcast_tensors(output, output_ref)
    torch.testing.assert_close(
        output,
        output_ref,
        atol=atol,
        rtol=rtol,
        equal_nan=True,
    )


def exact_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Exact comparison for bool/int outputs."""
    assert torch.equal(output, output_ref), "output does not exactly match reference"


class FixtureMeta(type):
    """Metaclass that makes Fixture subclasses usable as @decorators.

    Usage:
        class MyFixture(FixtureBase):
            PARAMS = [("a, b", [(1, 2), (3, 4)])]

        @MyFixture
        def test_something(a, b): ...
    """

    def __call__(cls, fn):
        for names, values in reversed(cls.PARAMS):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class FixtureBase(metaclass=FixtureMeta):
    """Base class for reusable parametrize decorators.

    Subclass and set PARAMS to a list of (names_str, values_list) tuples.
    - Single entry with multiple param names -> explicit combinations
    - Multiple entries each with one param name -> cross-product
    """
    PARAMS = []


class TestBase(ABC):
    """Abstract base class for op correctness testing.

    Subclass must implement gen_inputs() and ref_program().
    Provides check() for comparing op output against reference.
    """
    __test__ = False

    @abstractmethod
    def gen_inputs(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def ref_program(self, *inputs: Tuple[torch.Tensor]) -> Any:
        raise NotImplementedError

    def check(self,
              op,
              *inputs: Tuple[torch.Tensor],
              compare=None,
              atol: float = 1e-08,
              rtol: float = 1e-05) -> None:
        """Check the correctness of the op against ref_program.

        Args:
            op: The operator to test.
            *inputs: Input tensors.
            compare: Custom comparison callable(output, output_ref) or list of
                callables (one per output). Defaults to allclose_compare.
            atol: Absolute tolerance for default allclose_compare.
            rtol: Relative tolerance for default allclose_compare.
        """
        if compare is None:
            compare = partial(allclose_compare, atol=atol, rtol=rtol)

        op_name = op.__class__.__name__
        op_module = op.__class__.__module__

        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                _logger.warning("op=%s module=%s status=skip_oom", op_name, op_module)
                return
            raise e

        outputs_ref = _to_tuple(outputs_ref)

        with torch.no_grad():
            outputs = op(*inputs)

        outputs = _to_tuple(outputs)

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"

        # Compute error metrics before comparison (so we report even on failure).
        max_abs_err = 0.0
        for output, output_ref in zip(outputs, outputs_ref, strict=True):
            if output_ref is not None:
                err = (output.float() - output_ref.float()).abs().max().item()
                max_abs_err = max(max_abs_err, err)

        # Store for conftest hook to attach to pytest report.
        _check_result.op_name = op_name
        _check_result.op_module = op_module
        _check_result.max_abs_err = max_abs_err

        comparators = [compare] * len(outputs) if callable(compare) else list(compare)

        for output, output_ref, cmp in zip(outputs, outputs_ref, comparators, strict=True):
            if output_ref is not None:
                cmp(output, output_ref)

        _logger.info("op=%s module=%s status=pass max_abs_err=%.2e",
                      op_name, op_module, max_abs_err)
