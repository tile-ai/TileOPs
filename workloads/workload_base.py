"""Base classes for workload definitions shared between tests and benchmarks.

WorkloadBase defines the contract: gen_inputs() for input generation.
A class named for one op also carries that op's ref_program.
FixtureMeta / FixtureBase provide reusable pytest parametrize decorators.

Tolerances, check() and roofline numbers stay in tests/ and benchmarks/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar
from zlib import crc32

import torch

from tileops.manifest.primitives import WORKLOAD_SEED

_F = TypeVar("_F", bound=Callable[..., Any])


class WorkloadBase(ABC):
    """Abstract base for workload definitions (input generation + parameters).

    Subclass must implement gen_inputs(). A subclass named for one op also
    defines that op's ref_program; a subclass describing only an input shape
    leaves ref_program to its consumers.
    Used by both tests (via TestBase) and benchmarks (via BenchmarkBase).

    Tolerances, check() and roofline methods are decisions and belong to the
    consumer — not here.
    """

    @abstractmethod
    def gen_inputs(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def rng(self, tag: str = "", *, device: torch.device | str = "cpu") -> torch.Generator:
        """A generator private to this workload class and *tag*.

        The global stream the conftests seed makes a run reproduce the run
        before it. It does not make one tensor stable: every draw takes the
        next values in that stream, so adding a draw anywhere earlier moves
        every draw after it. An input that must not move that way — a page
        layout a benchmark compares numbers on, say — comes from here instead.

        The seed is derived from the class name and *tag*, so two workloads
        never draw the same tensor by accident. Each call returns a freshly
        seeded generator, so two calls to one ``gen_inputs`` return equal
        inputs.

        A draw on a CUDA tensor needs ``device="cuda"``: a generator only
        feeds draws on its own device.
        """
        seed = (WORKLOAD_SEED ^ crc32(f"{type(self).__name__}:{tag}".encode())) & 0xFFFFFFFF
        return torch.Generator(device=device).manual_seed(seed)


class CallWorkload(WorkloadBase):
    """The inputs of one manifest call, a workload row instantiated from the op's entry.

    ``gen_inputs()`` returns the call-time inputs in signature order, ``None`` where the call
    omits one; ``arguments()`` the op's constructor arguments.
    """

    def __init__(self, call: Any, device: "torch.device | str" = "cuda"):
        self.call = call
        self.device = device

    def gen_inputs(self) -> tuple[Any, ...]:
        tensors = self.call.materialize(self.device)
        return tuple(tensors[t] for t in self.call.signature.inputs)

    def arguments(self) -> dict[str, Any]:
        return self.call.arguments(self.call.materialize(self.device))


class RandnWorkload(WorkloadBase):
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

    def __call__(cls, fn: _F) -> _F:
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
