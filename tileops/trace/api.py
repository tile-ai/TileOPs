"""Public API for the in-kernel timeline trace tool.

``trace`` is a single namespace object. Import it and use four groups of methods:

- **Switch** — ``enable`` / ``disable`` / ``enabled`` / ``output``: turn tracing on
  for this process and choose where dumps land.
- **Annotate** — ``group`` / ``range`` / ``range_start`` / ``range_end`` / ``record``
  / ``dag``: mark up a kernel body. Markers are *always* emitted; a build either
  lowers them (traced) or strips them (zero cost).
- **Build** — ``out_idx`` / ``finalize`` (and the ``lower`` / ``strip`` primitives):
  call inside a ``@tilelang.jit`` builder so the right ``out_idx`` and marker
  handling apply automatically based on the switch.
- **Run & export** — ``run`` (one call: execute, dump, return outputs), or the
  ``decode`` / ``dump`` / ``export_html`` building blocks.

End-to-end::

    import functools, tilelang
    import tilelang.language as T
    from tileops.trace import trace

    @functools.lru_cache
    def build():
        @tilelang.jit(out_idx=trace.out_idx(1))      # +1 output (slots) when tracing
        def factory(block=128):
            @T.prim_func
            def main(a: T.Tensor(...), b: T.Tensor(...), c: T.Tensor(...)):
                with T.Kernel(...):
                    with trace.range("compute"):
                        ...
            return trace.finalize(main, max_events=1024)   # lower if on, else strip
        return factory

    trace.enable()                                     # dumps to debug/ (gitignored)
    compiled = build()(block=128)
    c = trace.run(compiled, (a, b), stem="my_kernel")  # writes debug/my_kernel.html
"""

import os

from . import passes
from .decode import decode as _decode
from .markers import (
    DEFAULT_LANE,
    AnnoToken,
    GroupScope,
    RangeScope,
    close_range,
    declare_flow,
    emit_instant,
    open_range,
)
from .record import MAX_EVENTS_DEFAULT
from .ui import export_timeline_html as _export_timeline_html

__all__ = ["trace"]


class _Trace:
    """The ``trace`` namespace (use the singleton ``trace``, do not instantiate).

    Stateless except for the process-local run switch (``enabled`` /
    ``output``); all annotation state lives in the per-build registry.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._output = "debug"

    # == Switch =============================================================

    @property
    def enabled(self) -> bool:
        """Whether tracing is on for this process (default ``False``).

        Example:
            >>> trace.enabled
            False
        """
        return self._enabled

    @property
    def output(self) -> str:
        """Output directory for dumped artifacts (default ``"debug"``).

        Example:
            >>> trace.output
            'debug'
        """
        return self._output

    def enable(self, output: str = "debug") -> None:
        """Turn tracing on for this process and set the output directory.

        Call once at startup, before any traced kernel is built — a builder reads
        ``enabled`` at build time and is typically cached.

        Args:
            output: Directory for dumped artifacts (created on first ``dump``).
                Defaults to ``"debug"`` (gitignored).

        Example:
            >>> trace.enable()              # dumps under debug/
            >>> trace.enable("out/traces")  # or a directory of your choosing
        """
        self._enabled = True
        self._output = output

    def disable(self) -> None:
        """Turn tracing off for this process.

        Example:
            >>> trace.disable()
        """
        self._enabled = False

    # == Annotate (inside a kernel body) ====================================

    def group(self, name: str, lead: int) -> GroupScope:
        """Declare the logical work-group enclosed markers belong to.

        Governs only *who records* (the elected writer is ``tx == lead``); the
        enclosed compute still runs on all threads.

        Args:
            name: Work-group name, interned to a stable group id.
            lead: Branch-baseline thread id of the electing writer.

        Returns:
            A ``with`` context manager.

        Example:
            >>> with trace.group("producer", lead=0):
            ...     with trace.range("tma"):
            ...         ...  # only thread 0 records
        """
        return GroupScope(name, lead)

    def range(self, name: str, lane: str = DEFAULT_LANE, payload=None) -> RangeScope:
        """Time a span: RANGE_BEGIN on enter, RANGE_END on exit.

        Args:
            name: Range name, interned to a stable event id.
            lane: Render sub-lane, interned dynamically (default ``"main"``).
            payload: Optional 32-bit i32 payload — a Python ``int`` or a runtime
                PrimExpr (e.g. a loop index). Records in both BEGIN and END events.
                Useful for explicitly tagging iterations in a loop. ``None`` defaults
                to 0, representing "no label". The payload is a user-provided tag,
                NOT an implicit value derived from thread/block indices.

        Returns:
            A ``with`` context manager.

        Example:
            >>> with trace.range("mma", lane="wgmma"):
            ...     T.wgmma_gemm(...)
            >>> # Explicitly tag each iteration with its index:
            >>> for i in range(N):
            ...     with trace.range("iteration", payload=i):
            ...         work()
        """
        return RangeScope(name, lane, payload)

    def range_start(self, name: str, lane: str = DEFAULT_LANE, payload=None) -> "AnnoToken":
        """Open a range explicitly (use when ``with`` does not fit the control flow).

        Args:
            name: Range name, interned to a stable event id.
            lane: Render sub-lane, interned dynamically (default ``"main"``).
            payload: Optional 32-bit i32 payload — a Python ``int`` or a runtime
                PrimExpr (e.g. a loop index). ``None`` defaults to 0, representing
                "no label". The payload is a user-provided tag, NOT an implicit
                value derived from thread/block indices.

        Returns:
            A token to pass to ``range_end``.

        Example:
            >>> tok = trace.range_start("phase")
            >>> ...  # work
            >>> trace.range_end(tok)
            >>> # Explicitly tag with payload:
            >>> tok = trace.range_start("iteration", payload=i)
            >>> ...
            >>> trace.range_end(tok)
        """
        return open_range(name, lane, payload)

    def range_end(self, tok: "AnnoToken | None") -> None:
        """Close the range opened for ``tok`` (``None`` no-ops).

        Args:
            tok: The token returned by ``range_start``.

        Example:
            >>> trace.range_end(tok)
        """
        close_range(tok)

    def record(self, name: str, payload=None, lane: str = DEFAULT_LANE) -> None:
        """Emit a single instant event (a zero-width mark).

        Args:
            name: Event name, interned to a stable event id.
            payload: Optional 32-bit i32 payload — a Python ``int`` or a runtime
                PrimExpr (e.g. a loop index). ``None`` records 0.
            lane: Render sub-lane, interned dynamically (default ``"main"``).

        Example:
            >>> trace.record("iter", payload=ki)  # tag the mark with the loop index
        """
        emit_instant(name, payload, lane)

    def dag(self, src_name: str, dst_name: str) -> None:
        """Declare a dependency arrow from one named range to another.

        Call once in the kernel body. Emits no device marker; at render time each
        CTA's ``src_name`` slices are paired with its ``dst_name`` slices in
        timestamp order, drawing one arrow per pair.

        Args:
            src_name: Source (producer-side) range name.
            dst_name: Destination (consumer-side) range name.

        Example:
            >>> trace.dag("arrive", "wait")  # producer "arrive" -> consumer "wait"
        """
        declare_flow(src_name, dst_name)

    # == Build (inside a @tilelang.jit builder) =============================

    def out_idx(self, n_outputs: int, traced: bool | None = None) -> list[int]:
        """Return the ``out_idx`` for a kernel with ``n_outputs`` real outputs.

        Grows by one (for the trailing ``slots`` output that ``finalize``
        appends) when traced, so the same builder works either way.

        Args:
            n_outputs: Number of real (non-``slots``) outputs.
            traced: Whether this build is traced. ``None`` (default) reads the
                process switch ``enabled``. **A cached builder must pass an
                explicit value** and include it in its cache key — otherwise the
                ``out_idx`` baked into the cached kernel ignores later switch flips.

        Returns:
            Negative output indices for ``@tilelang.jit``.

        Example:
            >>> @tilelang.jit(out_idx=trace.out_idx(1))          # follows the switch
            ... def factory(): ...
            >>> @tilelang.jit(out_idx=trace.out_idx(1, traced))  # cached builder
            ... def factory(): ...
        """
        enabled = self._enabled if traced is None else traced
        n = n_outputs + 1 if enabled else n_outputs
        return list(range(-n, 0))

    def finalize(self, primfunc, traced: bool | None = None,
                 max_events: int = MAX_EVENTS_DEFAULT):
        """Lower the markers when traced, else strip them — return either.

        The one line a builder returns instead of branching. Pairs with
        ``out_idx``; pass the same ``traced`` to both.

        Args:
            primfunc: The built kernel; its body carries ``trace.*`` markers.
            traced: Whether this build is traced. ``None`` (default) reads the
                process switch ``enabled``. A cached builder must pass an
                explicit value (see ``out_idx``).
            max_events: Per-slot event capacity when lowering.

        Returns:
            A ``PrimFunc`` ready for ``@tilelang.jit`` (with a trailing ``slots``
            output when traced).

        Example:
            >>> def factory():
            ...     @T.prim_func
            ...     def main(...): ...
            ...     return trace.finalize(main, traced, max_events=1024)
        """
        enabled = self._enabled if traced is None else traced
        return self.lower(primfunc, max_events) if enabled else self.strip(primfunc)

    def lower(self, primfunc, max_events: int = MAX_EVENTS_DEFAULT):
        """Primitive: materialize the markers and append the ``slots`` output.

        Prefer ``finalize`` unless you need to lower unconditionally.

        Args:
            primfunc: The built kernel; its body carries ``trace.*`` markers.
            max_events: Per-slot event capacity (compile-time cursor bound).

        Returns:
            The lowered ``PrimFunc`` with a trailing ``slots`` int64 output.

        Example:
            >>> return trace.lower(main, max_events=1024)
        """
        return passes.lower(primfunc, max_events=max_events)

    def strip(self, primfunc):
        """Primitive: no-op every marker so the kernel compiles without tracing.

        Prefer ``finalize`` unless you need to strip unconditionally. The
        generated CUDA is identical to an un-instrumented build.

        Args:
            primfunc: The built kernel; its body may carry ``trace.*`` markers.

        Returns:
            A ``PrimFunc`` with markers no-opped, signature unchanged.

        Example:
            >>> return trace.strip(main)
        """
        return passes.strip(primfunc)

    # == Run & export =======================================================

    def run(self, compiled, inputs: tuple, *, stem: str):
        """Run a kernel and, when tracing is on, dump its timeline.

        The one call a ``forward`` needs — it branches on the switch internally,
        so the caller does not. With tracing **off** it just returns the kernel's
        outputs unchanged. With tracing **on** the kernel is the traced build
        (returns ``(*real_outputs, slots)``): this splits the trailing ``slots``
        tensor off, decodes it, writes the timeline via ``dump`` (a fresh file
        each call), and returns the real outputs only.

        Args:
            compiled: A compiled kernel. Build it with ``traced=trace.enabled``
                (via ``out_idx`` / ``finalize``) so its outputs match the
                switch.
            inputs: Tuple of positional tensors to pass to ``compiled``.
            stem: Descriptive file stem (op name + shape); see ``dump``.

        Returns:
            The kernel's real outputs: a single tensor, or a tuple in order.

        Example:
            >>> compiled = build()(block=128)
            >>> c = trace.run(compiled, (a, b), stem="gemm_128x256x512")
        """
        out = compiled(*inputs)
        if not self._enabled:
            return out
        real = list(out) if isinstance(out, (tuple, list)) else [out]
        slots = real.pop()
        base = self.dump(self.decode(compiled, slots), compiled, stem=stem)
        print(f"[trace] wrote timeline {base}.html")
        return real[0] if len(real) == 1 else tuple(real)

    def decode(self, compiled, slots) -> list:
        """Decode a returned ``slots`` tensor into render-ready events.

        Resolves the host decode maps from the compiled kernel.

        Args:
            compiled: The compiled traced kernel (carries the host maps).
            slots: The trailing ``slots`` int64 tensor the kernel returned.

        Returns:
            A flat list of ``Slice`` / ``Instant`` events.

        Example:
            >>> *_, slots = compiled(a, b)
            >>> events = trace.decode(compiled, slots)
        """
        maps = passes.lookup_meta(compiled)
        return _decode(slots, id_to_name=maps["id_to_name"],
                       group_id_to_name=maps["group_id_to_name"],
                       lane_id_to_name=maps["lane_id_to_name"],
                       max_events=maps["max_events"], num_groups=maps["num_groups"])

    def dump(self, events: list, compiled, *, stem: str) -> str:
        """Write events as an HTML timeline under ``output``, never overwriting.

        Picks a collision-free base name: ``{output}/{stem}`` when free, else
        ``{output}/{stem}_1``, ``_2``, ... so repeated dumps accumulate.

        Args:
            events: Event list from ``decode``.
            compiled: The compiled traced kernel (carries the host maps).
            stem: File stem (no directory, no extension), e.g. op name + shape.

        Returns:
            The base path written (no extension); ``{base}.html`` exists.

        Example:
            >>> base = trace.dump(events, compiled, stem="gemm_128x256x512")
            >>> base
            'debug/gemm_128x256x512'
        """
        os.makedirs(self._output, exist_ok=True)
        base = os.path.join(self._output, stem)
        if os.path.exists(f"{base}.html"):
            i = 1
            while os.path.exists(f"{base}_{i}.html"):
                i += 1
            base = f"{base}_{i}"
        self.export_html(events, f"{base}.html", compiled=compiled)
        return base

    def export_html(self, events: list, path: str, *, compiled,
                    title: str = "", sm_clock_ghz: float = 1.5) -> None:
        """Write events as a self-contained Plotly HTML timeline.

        Args:
            events: Event list from ``decode``.
            path: Destination ``.html`` path.
            compiled: The compiled traced kernel (carries the host maps).
            title: Base figure title; the CTA index is appended per tab.
            sm_clock_ghz: Locked SM clock in GHz for the ``~ns`` hover column.

        Example:
            >>> trace.export_html(events, "timeline.html", compiled=compiled)
        """
        maps = passes.lookup_meta(compiled)
        _export_timeline_html(events, path, group_id_to_name=maps["group_id_to_name"],
                              lane_id_to_name=maps["lane_id_to_name"],
                              flows=maps["flows"], title=title,
                              sm_clock_ghz=sm_clock_ghz)


# The singleton namespace: ``from tileops.trace import trace``.
trace = _Trace()
