"""Base classes for workload definitions shared between tests and benchmarks.

WorkloadBase defines the contract: gen_inputs() for input generation.
FixtureMeta / FixtureBase provide reusable pytest parametrize decorators.

Correctness-only logic (ref_program, check, tolerances) stays in tests/.
"""

from abc import ABC, abstractmethod
from typing import Any

import pytest


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
