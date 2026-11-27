"""Frontend ``trace.lower`` transform: materialize trace placeholders.

Rewrites ``tl_trace_marker`` placeholders from the trace annotation API into
event writes and appends a ``slots`` output to the ``PrimFunc`` before JIT compilation.

slots buffer layout (the contract decode reads against)
--------------------------------------------------------
One injected output param, the LAST output::

    slots : T.Tensor((num_cta, num_groups, (max_events + 1) * 2), "int64")

Per ``(cta, gid)`` slot (a flat row of ``(max_events + 1) * 2`` int64 words):

* ``word[0]``  = event count (header), saturated at ``max_events``.
* ``word[1]``  = reserved (unused garbage; decoders ignore it).
* event ``e`` in ``[0, max_events)`` occupies words ``[(e + 1) * 2, (e + 1) * 2 + 1]``::

      word[(e + 1) * 2]      = w0 = clock64() timestamp (int64)
      word[(e + 1) * 2 + 1]  = w1 = pack_w1(event_id, kind, lane, payload)

NO-STOMP guarantee: the write offset uses ``idx = min(cursor, max_events - 1)``
and the header count saturates with ``count = min(cursor + 1, max_events)``, so a
writer can never address a word past its own slot's event region — this holds
even though TileLang's simplifier collapses a plain ``if cursor < max_events``
guard. The clamp, not the guard, is the contract.

Header zeroing: the output buffer is adapter-allocated via ``torch.empty`` (not
zeroed), so the elected writer of each group zeroes its header ``word[0]`` in the
kernel prologue before any marker fires.

Host maps: ``lower`` stores each build's decode maps in ``_META_REGISTRY`` under an id
stamped on the returned ``PrimFunc`` as the ``tl.trace_meta_id`` attr, which
``tilelang.jit`` carries onto ``kernel.prim_func``; ``lookup_meta`` reads it back.
"""

from pathlib import Path

# Importing tilelang first populates sys.path for the bundled tvm package.
import tilelang  # noqa: F401  (loads tvm before the tvm imports below)
import tilelang.language as T
import tvm.tirx as tx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit

from tileops._csrc import csrc_path
from tileops.trace._record import MAX_EVENTS_DEFAULT, pack_w1_tir
from tileops.trace._state import MARKER, begin_build_epoch, build_state

__all__ = ["is_traced", "lookup_meta", "lower", "strip"]

_HELPER = Path(csrc_path("trace_helper.h")).read_text()

_META_ATTR = "tl.trace_meta_id"
_META_REGISTRY: dict[int, dict] = {}
_NEXT_META_ID = 0


def _is_marker(node) -> bool:
    """True if ``node`` is a ``tl_trace_marker`` placeholder Call (StringImm arg0)."""
    if not isinstance(node, tx.Call):
        return False
    args = list(node.args)
    return bool(args) and isinstance(args[0], tx.StringImm) and args[0].value == MARKER


def _transform(primfunc, max_events: int, num_groups: int, lead_fn):
    """Rewrite each placeholder into writer-gated, clamped event writes and append the
    ``slots`` param (layout in the module docstring).

    The CTA dimension is the launch grid's size and a CTA's row its hardware block
    index. ``lead_fn`` maps a gid to the thread that writes its markers. Returns the new
    ``PrimFunc`` and the CTA count.
    """
    # Header words per slot: word[0] = count, word[1] = reserved. Events follow.
    header_words = 2
    slot_words = header_words + 2 * max_events

    # Launch extents must be concrete IntImm values at transform time.
    bind = {}

    def collect(n):
        if isinstance(n, tx.For) and n.thread_binding is not None:
            bind[n.thread_binding.thread_tag] = n

    post_order_visit(primfunc.body, collect)

    # T.Kernel(..., threads=N) binds no threadIdx.x; _HELPER's tileops::trace_thread_idx_x() stands
    # in for it when electing the thread that writes markers.
    if "threadIdx.x" not in bind:
        tx_ = T.call_extern("int32", "tileops::trace_thread_idx_x")
    else:
        tx_ = bind["threadIdx.x"].loop_var

    # A CTA's slot row is its hardware block index, read by _HELPER's
    # tileops::trace_cta_index(). The blockIdx loop vars cannot name it: a rasterized
    # kernel (T.use_swizzle) rebinds blockIdx after the prologue that zeroes the header,
    # so the header and the events would land in different rows.
    cta_flat = T.call_extern("int32", "tileops::trace_cta_index")
    num_cta = 1
    for tag in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
        if tag in bind:
            num_cta *= int(bind[tag].extent)

    slots_buf = tx.decl_buffer(
        (num_cta, num_groups, slot_words), "int64", name="slots", scope="global"
    )
    slots_var = tx.Var("slots_handle", "handle")

    # A global header cursor avoids constant folding of a local cursor and its
    # guards. Each group has one writer, so the update needs no atomics.
    def emit_for_marker(call):
        # event_id / kind / lane / gid are compile-time-constant interned ids;
        # payload is the one runtime arg, kept as-is for pack_w1_tir.
        event_id, kind, lane, gid = (int(a) for a in call.args[1:5])
        payload_expr = call.args[5]
        is_writer = tx_ == lead_fn(gid)
        i = tx.Cast("int32", tx.BufferLoad(slots_buf, [cta_flat, gid, 0]))
        in_room = i < max_events
        # No-stomp #1: hard-clamp the write index to the last legal event slot
        # so a collapsed guard can never address past this slot's event region.
        idx = tx.Min(i, max_events - 1)
        base = header_words + idx * 2
        ts = T.call_extern("uint64", "tileops::trace_now")
        w1 = pack_w1_tir(event_id, kind, lane, payload_expr)
        store_w0 = tx.BufferStore(slots_buf, tx.Cast("int64", ts), [cta_flat, gid, base])
        store_w1 = tx.BufferStore(slots_buf, w1, [cta_flat, gid, base + 1])
        # No-stomp #2: saturate the header count so the decoder never reads more
        # events than the slot physically holds.
        new_cnt = tx.Min(i + 1, max_events)
        store_cnt = tx.BufferStore(slots_buf, tx.Cast("int64", new_cnt), [cta_flat, gid, 0])
        body = tx.SeqStmt([store_w0, store_w1, store_cnt])
        guarded = tx.IfThenElse(in_room, body, None)
        return tx.IfThenElse(is_writer, guarded, None)

    def post(node):
        if isinstance(node, tx.Evaluate) and _is_marker(node.value):
            return emit_for_marker(node.value)
        if isinstance(node, tx.SBlock) and node.name_hint == "tilelang_root":
            anns = dict(node.annotations) if node.annotations else {}
            anns["pragma_import_c"] = _HELPER
            # torch.empty output is NOT pre-zeroed; each group's elected writer
            # zeroes its header count before any marker fires.
            zero_stores = []
            for g in range(num_groups):
                lead_g = lead_fn(g)
                zero = tx.BufferStore(slots_buf, tx.IntImm("int64", 0), [cta_flat, g, 0])
                zero_stores.append(tx.IfThenElse(tx_ == lead_g, zero, None))
            zero_body = zero_stores[0] if len(zero_stores) == 1 else tx.SeqStmt(zero_stores)
            new_body = tx.SeqStmt([zero_body, node.body])
            return tx.SBlock(
                iter_vars=node.iter_vars,
                reads=node.reads,
                writes=node.writes,
                name_hint=node.name_hint,
                body=new_body,
                init=node.init,
                alloc_buffers=node.alloc_buffers,
                match_buffers=node.match_buffers,
                annotations=anns,
            )
        return node

    new_body = ir_transform(primfunc.body, None, post)

    new_bmap = dict(primfunc.buffer_map)
    new_bmap[slots_var] = slots_buf
    lowered = tx.PrimFunc(
        params=list(primfunc.params) + [slots_var],
        body=new_body,
        ret_type=primfunc.ret_type,
        buffer_map=new_bmap,
        attrs=primfunc.attrs,
    )
    return lowered, num_cta


def lower(primfunc, max_events: int = MAX_EVENTS_DEFAULT):
    """Lower the markers in ``primfunc`` and append the ``slots`` int64 output param.

    The builder's ``@tilelang.jit`` ``out_idx`` must count the trailing ``slots``. The
    decode maps are registered under the ``tl.trace_meta_id`` attr set on the result, and
    the build epoch is retired.
    """
    global _NEXT_META_ID

    state = build_state()
    begin_build_epoch()

    num_groups = max(len(state.group_id_to_name), 1)

    def lead_fn(gid: int) -> int:
        return state.group_id_to_lead.get(gid, 0)

    lowered, num_cta = _transform(primfunc, max_events, num_groups, lead_fn)

    host_maps = {
        "id_to_name": dict(state.id_to_name),
        "group_id_to_name": dict(state.group_id_to_name),
        "group_id_to_lead": dict(state.group_id_to_lead),
        "lane_id_to_name": dict(state.lane_id_to_name),
        "flows": list(state.declared_flows),
        "max_events": max_events,
        "num_groups": num_groups,
        "num_cta": num_cta,
    }

    meta_id = _NEXT_META_ID
    _NEXT_META_ID += 1
    _META_REGISTRY[meta_id] = host_maps
    return lowered.with_attr(_META_ATTR, meta_id)


def strip(primfunc):
    """Replace every marker with a no-op so ``primfunc`` compiles without tracing.

    The result has no ``slots`` param and the generated CUDA is identical to an
    un-instrumented build. Like ``lower``, it retires the build epoch, so the names this
    build interned do not reach the next one.
    """
    begin_build_epoch()

    def post(node):
        if isinstance(node, tx.Evaluate) and _is_marker(node.value):
            return tx.Evaluate(tx.IntImm("int32", 0))
        return node

    return tx.PrimFunc(
        params=list(primfunc.params),
        body=ir_transform(primfunc.body, None, post),
        ret_type=primfunc.ret_type,
        buffer_map=dict(primfunc.buffer_map),
        attrs=primfunc.attrs,
    )


def _meta_id(kernel) -> "int | None":
    """The ``tl.trace_meta_id`` ``lower`` stamped on *kernel*'s ``prim_func``, or ``None``."""
    attrs = getattr(getattr(kernel, "prim_func", None), "attrs", None)
    meta_id = attrs.get(_META_ATTR) if attrs is not None else None
    return None if meta_id is None else int(meta_id)


def is_traced(kernel) -> bool:
    """Whether *kernel* was built from ``lower``'s result, so its last output is ``slots``."""
    return _meta_id(kernel) is not None


def lookup_meta(kernel) -> dict:
    """The host decode maps ``lower`` registered for *kernel*.

    Raises ``ValueError`` when *kernel* was not built from ``lower``'s result.
    """
    meta_id = _meta_id(kernel)
    if meta_id is None:
        raise ValueError(
            "kernel was not produced by trace.lower (no tl.trace_meta_id attr on "
            "kernel.prim_func); cannot resolve host decode maps"
        )
    if meta_id not in _META_REGISTRY:
        raise ValueError(f"unknown trace meta id {meta_id}; host maps not registered")
    return _META_REGISTRY[meta_id]
