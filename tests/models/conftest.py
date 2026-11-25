"""Fixtures the model tests share."""

import collections

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
