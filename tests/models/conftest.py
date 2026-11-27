"""Fixtures and helpers the model tests share."""

import collections
from collections.abc import Callable
from typing import Any

import pytest
import torch


@pytest.fixture
def graph_calls(monkeypatch) -> collections.Counter:
    """How many CUDA graphs the test captures (``capture_end``) and replays (``replay``)."""
    calls = collections.Counter()
    for name in ("capture_end", "replay"):
        method = getattr(torch.cuda.CUDAGraph, name)

        def counted(graph, *args, method=method, name=name):
            calls[name] += 1
            return method(graph, *args)

        monkeypatch.setattr(torch.cuda.CUDAGraph, name, counted)
    return calls


class Recorded:
    """Wraps a model part: each call returns what the part returns, after passing the result
    and the call's arguments to ``record``."""

    def __init__(self, part: Any, record: Callable[..., None]) -> None:
        self._part = part
        self._record = record

    def __getattr__(self, name: str) -> Any:
        return getattr(self._part, name)

    def __call__(self, *args: Any) -> Any:
        out = self._part(*args)
        self._record(out, *args)
        return out
