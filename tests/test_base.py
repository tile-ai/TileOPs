from abc import ABC, abstractmethod
from typing import Any, Tuple

import pytest
import torch


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
              atol: float = 1e-08,
              rtol: float = 1e-05) -> None:
        """Check the correctness of the op against ref_program."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                max_err = (output - output_ref).abs().max()
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {max_err}"

        print(f"All checks passed for {op.__class__.__name__}.")

    def check_fn(self,
                 fn: callable,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-08,
                 rtol: float = 1e-05,
                 grad: bool = True) -> None:
        """Check the correctness of a function (with optional backward pass)."""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        if not grad:
            with torch.no_grad():
                outputs = fn(*inputs)
        else:
            output = fn(*inputs)
            loss = output.sum()
            loss.backward()
            outputs = []
            outputs.append(output)
            for inp in inputs:
                outputs.append(inp.grad)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(
                zip(outputs, outputs_ref, strict=True)):
            if output_ref is not None:
                max_err = (output - output_ref).abs().max()
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {max_err}"

        print(f"All checks passed for {fn.__class__.__name__}.")
