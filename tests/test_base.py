import logging
import threading
from abc import abstractmethod
from functools import partial
from typing import Any

import torch

from workloads.workload_base import FixtureBase, FixtureMeta, WorkloadBase

_logger = logging.getLogger("tileops.ops")

# Thread-local storage for conftest hook to pick up per-test Op info.
_check_result = threading.local()

# Re-export for backward compatibility with tests that import from here.
__all__ = [
    "FixtureBase",
    "FixtureMeta",
    "TestBase",
    "allclose_compare",
    "exact_compare",
]


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


class TestBase(WorkloadBase):
    """Abstract base class for op correctness testing.

    Inherits gen_inputs() from WorkloadBase.
    Subclasses must implement ref_program() for correctness checking.
    Provides check() for comparing op output against reference.
    """
    __test__ = False

    @abstractmethod
    def ref_program(self, *inputs: Any) -> Any:
        """Reference implementation for correctness checking.

        Must be overridden by every concrete test subclass.
        Should return the same outputs as the op under test, using a
        simpler/trusted implementation (e.g. PyTorch built-ins).
        """
        raise NotImplementedError

    def check(self,
              op,
              *inputs: torch.Tensor,
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
