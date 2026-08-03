"""Base classes for workload definitions shared between tests and benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from utils import device_str


class WorkloadBase(ABC):
    """Abstract base for workload definitions (input generation + parameters).

    Subclass must implement gen_inputs().
    Used by both tests (correctness) and benchmarks (profiling).
    """

    @abstractmethod
    def gen_inputs(self) -> tuple[Any, ...]:
        raise NotImplementedError

    @staticmethod
    def make_tensor(shape, dtype, *, randn=True) -> torch.Tensor:
        if randn:
            return torch.randn(*shape, dtype=dtype, device=device_str())
        return torch.zeros(*shape, dtype=dtype, device=device_str())
