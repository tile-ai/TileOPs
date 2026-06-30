"""In-kernel timeline trace tool: annotate, build, run, and export a timeline.

The whole surface is the :data:`trace` namespace (see :mod:`tileops.trace.api`
for the full, example-driven reference). In short: ``trace.enable()`` turns
tracing on, ``with trace.range(...)`` / ``trace.group(...)`` mark up a kernel
body, ``trace.out_idx`` + ``trace.finalize`` wire it into a vanilla
``@tilelang.jit`` builder, and ``trace.run`` executes the kernel and dumps the
timeline. With tracing off, markers are stripped and the generated CUDA is
identical to an un-instrumented build.

Lane names are interned dynamically at build time (default ``"main"``); at most
16 distinct lanes fit the packed 4-bit lane field.

Importing this package does **not** touch ``tilelang``: there is no monkeypatch,
and ``tilelang.compile`` / ``tilelang.jit.compile`` stay the originals.
"""

from .api import trace
from .decode import FlowEdge, Instant, Slice, compute_flows, decode
from .record import MAX_EVENTS_DEFAULT, EventKind, pack_w1, unpack_w1
from .ui import export_timeline_html

__all__ = [
    "MAX_EVENTS_DEFAULT",
    "EventKind",
    "FlowEdge",
    "Instant",
    "Slice",
    "compute_flows",
    "decode",
    "export_timeline_html",
    "pack_w1",
    "trace",
    "unpack_w1",
]
