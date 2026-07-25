"""Contract tests for :func:`benchmarks.benchmark_base._native_output_suppressor`.

``bench_kernel`` silences C++-level profiler chatter with an fd-level
redirect (tilelang's ``suppress_stdout_stderr``). That redirect targets
``sys.stdout.fileno()`` / ``sys.stderr.fileno()``; when a test harness has
already replaced those streams (pytest fd capture), redirecting the
underlying descriptors corrupts the harness's capture stream. The helper
must therefore return the fd-level suppressor only when stdout/stderr still
point at the process-level descriptors 1 and 2, and a no-op context
otherwise.
"""

from __future__ import annotations

import contextlib
import io
import sys

import pytest

from benchmarks.benchmark_base import _native_output_suppressor

pytestmark = pytest.mark.smoke


class _FakeStream:
    """Stream stub reporting a fixed underlying file descriptor."""

    def __init__(self, fd: int):
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


def test_suppresses_when_streams_are_process_fds(monkeypatch):
    """Standalone runs (stdout->fd 1, stderr->fd 2) get the fd-level
    suppressor. The context is not entered: doing so would redirect the
    real process fds."""
    monkeypatch.setattr(sys, "stdout", _FakeStream(1))
    monkeypatch.setattr(sys, "stderr", _FakeStream(2))

    ctx = _native_output_suppressor()
    assert not isinstance(ctx, contextlib.nullcontext)


def test_noop_when_streams_are_redirected(monkeypatch):
    """Captured runs (fileno is a capture tmpfile, not fd 1/2) must get a
    no-op context so the capture descriptors are never clobbered."""
    monkeypatch.setattr(sys, "stdout", _FakeStream(7))
    monkeypatch.setattr(sys, "stderr", _FakeStream(8))

    ctx = _native_output_suppressor()
    assert isinstance(ctx, contextlib.nullcontext)


def test_noop_when_streams_have_no_fileno(monkeypatch):
    """Streams without a real descriptor (e.g. ``io.StringIO`` under
    ``capsys``) raise from ``fileno()``; the helper must degrade to a
    no-op context instead of propagating."""
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    ctx = _native_output_suppressor()
    assert isinstance(ctx, contextlib.nullcontext)
