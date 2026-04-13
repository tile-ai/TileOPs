"""Base classes for workload definitions shared between tests and benchmarks.

WorkloadBase defines the contract: gen_inputs() for input generation.
FixtureMeta / FixtureBase provide reusable pytest parametrize decorators.

Correctness-only logic (ref_program, check, tolerances) stays in tests/.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class WorkloadBase(ABC):
    """Abstract base for workload definitions (input generation + parameters).

    Subclass must implement gen_inputs().
    Used by both tests (via TestBase) and benchmarks (via BenchmarkBase).

    Correctness-only methods (ref_program, check, tolerances) belong in
    tests/ — not here.
    """

    @abstractmethod
    def gen_inputs(self) -> Any:
        raise NotImplementedError


class RandnTest(WorkloadBase):
    """Workload base for ops whose inputs are generated via ``torch.randn``."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)


class FixtureMeta(type):
    """Metaclass that makes Fixture subclasses usable as @decorators.

    Usage:
        class MyFixture(FixtureBase):
            @classmethod
            def get_params(cls):
                import pytest
                return [("a, b", [
                    pytest.param(1, 2, marks=pytest.mark.smoke),
                ])]

        @MyFixture
        def test_something(a, b): ...

    PARAMS may also be set as a plain class variable (list) for backwards
    compatibility when pytest is already importable at module scope.
    """

    def __call__(cls, fn):
        import pytest  # lazy import: pytest is only needed when applying parametrize decorators

        params = cls.get_params() if hasattr(cls, "get_params") else cls.PARAMS
        for names, values in reversed(params):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class FixtureBase(metaclass=FixtureMeta):
    """Base class for reusable parametrize decorators.

    Subclass and set PARAMS (plain list) or override get_params() (classmethod
    that lazily imports pytest) to provide a list of (names_str, values_list)
    tuples.
    - Single entry with multiple param names -> explicit combinations
    - Multiple entries each with one param name -> cross-product
    """
    PARAMS = []
