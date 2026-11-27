"""Base classes of the workloads tests and benchmarks share, and reusable pytest
parametrize decorators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar
from zlib import crc32

import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import entry_plan
from tileops.manifest._workload import Call, instantiate
from tileops.manifest.primitives import WORKLOAD_SEED
from workloads.device import run_device
from workloads.numerics import Evidence, Exact

_F = TypeVar("_F", bound=Callable[..., Any])


class WorkloadBase(ABC):
    """Inputs of an op call (``gen_inputs``), its reference and its numerical declaration.

    A subclass named for one op defines ``ref_program``: float32 or wider, rounded once at
    its output, earlier only where the op's contract stores an intermediate. An optional
    ``torch_program`` is the op as torch runs it at the input dtype. ``verification()`` owns
    numerical policy; consumers own scheduling and reporting.
    """

    # Whether ref_program reads an op call's inputs in the form its manifest call states.
    # False where gen_inputs makes another form that the workload's benchmark adapts for
    # the op; a consumer passing an op call's own inputs then has no reference for them.
    reference_takes_call_inputs = True

    @abstractmethod
    def gen_inputs(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def verification(self, *inputs: Any) -> Evidence:
        """The correctness declaration shared by tests and every benchmark tag."""
        return Exact()

    def rng(self, tag: str = "", *, device: torch.device | str = "cpu") -> torch.Generator:
        """A freshly seeded generator private to this workload class and *tag*, on *device*.

        An input drawn from it does not move when a draw is added anywhere earlier in the
        global stream; draws on a device tensor need that *device*.
        """
        seed = (WORKLOAD_SEED ^ crc32(f"{type(self).__name__}:{tag}".encode())) & 0xFFFFFFFF
        return torch.Generator(device=device).manual_seed(seed)


def manifest_call(op: str, dtype_case: dict | None = None, **row) -> Call:
    """The manifest call of *op* that *row* describes, as a workload row would.

    A test picks its own shapes; the call still takes its metadata from the entry's
    generators and its contract from the entry's signature (docs/design/manifest.md § Rows).
    """
    plan = entry_plan(op, load_manifest()[op], load_adts(), resolve=False)
    return instantiate(plan, {**row, "label": "test"}, dtype_case or {})


class CallWorkload(WorkloadBase):
    """The inputs of one manifest call, a workload row instantiated from the op's entry.

    ``gen_inputs()`` returns the call-time inputs in signature order, ``None`` where the call
    omits one; ``arguments()`` the op's constructor arguments.
    """

    def __init__(self, call: Any, device: torch.device | str | None = None):
        self.call = call
        self._device = device

    @property
    def device(self) -> torch.device | str:
        """The device the inputs go on: the one passed in, else the run's."""
        return self._device if self._device is not None else run_device()

    def gen_inputs(self) -> tuple[Any, ...]:
        tensors = self.call.materialize(self.device)
        return tuple(tensors[t] for t in self.call.signature.inputs)

    def arguments(self) -> dict[str, Any]:
        return self.call.arguments()


class FixtureMeta(type):
    """Makes a fixture class a decorator applying ``pytest.mark.parametrize`` once per
    ``(names, values)`` entry of its ``PARAMS``, or of ``get_params()`` where it defines one."""

    def __call__(cls, fn: _F) -> _F:
        # Imported here so that importing workloads does not need pytest.
        import pytest

        params = cls.get_params() if hasattr(cls, "get_params") else cls.PARAMS
        for names, values in reversed(params):
            fn = pytest.mark.parametrize(names, values)(fn)
        return fn


class FixtureBase(metaclass=FixtureMeta):
    """A reusable parametrize decorator: ``PARAMS`` lists ``(names, values)`` entries, and
    entries stack as a cross product."""

    PARAMS = []
