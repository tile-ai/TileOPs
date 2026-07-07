"""Frontend ``trace.lower`` transform: materialize trace placeholders.

This is the build-time half of the trace tool. It pairs with the module-level
``tileops.trace.api.trace`` annotation API: that API emits zero-cost
``tl_trace_marker`` placeholders into a kernel body as it is built; ``trace.lower``
then rewrites every placeholder into real emit code and appends a ``slots`` output
param, returning a plain ``PrimFunc`` that a *vanilla* ``@tilelang.jit`` compiles.

There is no monkeypatch of ``tilelang``: importing this module leaves
``tilelang.compile`` / ``tilelang.jit.compile`` untouched. The whole surface is
the explicit ``lower`` transform.

Module name note:
    ``pass`` is a Python keyword, so the file is named ``passes.py`` to stay a
    normal importable submodule.

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

Host-map registry
------------------
``trace.lower`` stashes the host-side decode maps in the module-level
``_META_REGISTRY`` dict, keyed by an integer meta id stamped onto the
returned ``PrimFunc`` as the ``tl.trace_meta_id`` attr. The attr rides through
``tilelang.jit`` onto ``kernel.prim_func``, so ``lookup_meta`` resolves a
compiled kernel back to its maps without relying on object identity.
"""

# Importing tilelang first populates sys.path for the bundled tvm package.
import tilelang  # noqa: F401  (loads tvm before the tvm imports below)
import tilelang.language as T  # noqa: F401
import tvm.tirx as tx
from tvm.tirx.stmt_functor import ir_transform, post_order_visit

from .device import _HELPER
from .record import MAX_EVENTS_DEFAULT, pack_w1_tir
from .state import MARKER, begin_build_epoch, build_state

__all__ = ["MAX_EVENTS_DEFAULT", "lookup_meta", "lower", "strip"]

# Header words per slot: word[0] = count, word[1] = reserved. Events follow.
HEADER_WORDS = 2

# PrimFunc attr that carries the host-map registry key onto the lowered func; it
# rides through tilelang.jit onto kernel.prim_func so lookup_meta resolves a
# compiled kernel back to its decode maps.
_META_ATTR = "tl.trace_meta_id"

# Module-level host-map registry: meta_id (int) -> host_maps dict. Populated by
# lower(), read by lookup_meta().
_META_REGISTRY: dict[int, dict] = {}
_NEXT_META_ID = 0


def _is_marker(node) -> bool:
    """True if ``node`` is a ``tl_trace_marker`` placeholder Call (StringImm arg0)."""
    if not isinstance(node, tx.Call):
        return False
    args = list(node.args)
    return bool(args) and isinstance(args[0], tx.StringImm) and args[0].value == MARKER


def _transform(primfunc, max_events: int, num_groups: int, lead_fn):
    """Rewrite trace placeholders into emit code; append the ``slots`` output param.

    See the module docstring for the slots layout and the no-stomp clamp
    contract. The flat CTA count is derived from the kernel's launch config (the
    product of the blockIdx extents); the buffer's CTA dimension uses it.

    Args:
        primfunc: The built kernel; body may contain ``tl_trace_marker`` markers.
        max_events: Compile-time per-slot event capacity (cursor bound).
        num_groups: Logical groups per CTA (gid range).
        lead_fn: ``gid -> lead`` lookup electing each group's writer (``tx == lead``).

    Returns:
        A ``(primfunc, num_cta)`` pair: the new ``PrimFunc`` with a trailing
        ``slots`` int64 param and every marker rewritten into writer-gated,
        clamped emit code, plus the derived flat CTA count.
    """
    slot_words = (max_events + HEADER_WORDS) * 2

    # 1) Collect thread/block binding For nodes (the loop var IS the blockIdx /
    #    threadIdx, the For extent IS the grid / block dimension). Every blockIdx
    #    and threadIdx axis appears, even when its extent is 1; the extent is a
    #    concrete IntImm at transform time (build-time-known launch config).
    bind = {}

    def collect(n):
        if isinstance(n, tx.For) and n.thread_binding is not None:
            bind[n.thread_binding.thread_tag] = n

    post_order_visit(primfunc.body, collect)
    tx_ = bind["threadIdx.x"].loop_var

    # Flatten the full grid: read each present blockIdx axis's var + extent and
    # build cta_flat = bx + by*gx + bz*gx*gy (only the axes that exist), with
    # num_cta = gx * gy * gz. A multi-dim grid (gy/gz > 1) would otherwise collide
    # every block sharing a (bx) row on the same slot.
    gx = gy = gz = 1
    cta_flat = tx.IntImm("int32", 0)
    stride = 1
    for tag in ("blockIdx.x", "blockIdx.y", "blockIdx.z"):
        node = bind.get(tag)
        if node is None:
            continue
        extent = int(node.extent)
        cta_flat = cta_flat + node.loop_var * stride if stride != 1 else node.loop_var
        stride *= extent
        if tag == "blockIdx.x":
            gx = extent
        elif tag == "blockIdx.y":
            gy = extent
        else:
            gz = extent
    num_cta = gx * gy * gz

    # 2) New slots param Var + Buffer: [num_cta, num_groups, slot_words] int64.
    slots_buf = tx.decl_buffer((num_cta, num_groups, slot_words), "int64",
                               name="slots", scope="global")
    slots_var = tx.Var("slots_handle", "handle")

    # 3) Cursor lives in the slot HEADER word (slots[cta,gid,0]), read-modify-write
    #    per marker. A global header RMW stays opaque to the simplifier (a local
    #    int32 cursor gets const-folded and its guards collapse). Single writer
    #    (tx==lead) => no atomics.
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
        base = HEADER_WORDS + idx * 2
        ts = T.call_extern("uint64", "__tl_now")
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

    # 4) Rewrite pass: replace each Evaluate(marker) with emit code; on the root
    #    SBlock inject the helper + zero each group's header word (writer only).
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
                iter_vars=node.iter_vars, reads=node.reads, writes=node.writes,
                name_hint=node.name_hint, body=new_body, init=node.init,
                alloc_buffers=node.alloc_buffers, match_buffers=node.match_buffers,
                annotations=anns,
            )
        return node

    new_body = ir_transform(primfunc.body, None, post)

    # 5) Append the slots param + buffer_map.
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


def _strip_markers(primfunc):
    """Drop every ``tl_trace_marker`` placeholder from ``primfunc`` (zero-cost path).

    The always-emit markers are no-opped before codegen so the generated CUDA is
    byte-identical to an un-instrumented build (no ``slots`` param, no
    ``__tl_now``). Backs the public ``strip`` escape hatch.

    Args:
        primfunc: A built kernel whose body may contain marker placeholders.

    Returns:
        A ``PrimFunc`` with every ``Evaluate(tl_trace_marker)`` replaced by a
        no-op ``Evaluate(0)``; unchanged in every other respect.
    """

    def post(node):
        if isinstance(node, tx.Evaluate) and _is_marker(node.value):
            return tx.Evaluate(tx.IntImm("int32", 0))
        return node

    new_body = ir_transform(primfunc.body, None, post)
    return tx.PrimFunc(
        params=list(primfunc.params),
        body=new_body,
        ret_type=primfunc.ret_type,
        buffer_map=dict(primfunc.buffer_map),
        attrs=primfunc.attrs,
    )


def lower(primfunc, max_events: int = MAX_EVENTS_DEFAULT):
    """Lower the always-emit markers in ``primfunc`` and append the ``slots`` output.

    The explicit build-time transform: call it inside a builder, return its
    result, and decorate the builder with a *vanilla* ``@tilelang.jit`` whose
    ``out_idx`` already accounts for the trailing ``slots`` param (e.g.
    ``out_idx=[-2, -1]`` when the kernel has one own output ``C`` and ``slots``).

    Reads the per-build trace registry (``tileops.trace.state.build_state``),
    derives ``num_groups`` from the distinct interned groups, runs
    ``_transform`` to rewrite each marker into writer-gated clamped emit code
    (deriving ``num_cta`` from the launch grid), then stashes the host decode maps
    in ``_META_REGISTRY`` under a fresh integer meta id and stamps that id
    onto the returned func as the ``tl.trace_meta_id`` attr. The attr rides
    through ``tilelang.jit`` onto ``kernel.prim_func`` so ``lookup_meta``
    resolves the compiled kernel back to its maps. It then retires the build epoch
    (``tileops.trace.state.begin_build_epoch``) so the next build starts
    fresh.

    Args:
        primfunc: The built kernel; its body carries ``tl_trace_marker`` markers.
        max_events: Per-slot event capacity (compile-time cursor bound). Defaults
            to ``tileops.trace.record.MAX_EVENTS_DEFAULT``.

    Returns:
        The lowered ``PrimFunc`` with the trailing ``slots`` int64 output param
        and the ``tl.trace_meta_id`` attr set. A vanilla ``@tilelang.jit`` over a
        builder returning this func compiles it; ``extract_params`` then sees
        ``slots`` as the last param.
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
    """No-op the always-emit markers so ``primfunc`` compiles WITHOUT tracing.

    The escape hatch: a kernel written with ``trace.*`` markers can be compiled
    zero-cost by passing it through ``strip`` instead of ``lower``. The
    returned func has no ``slots`` param and emits no ``__tl_now``, so the
    generated CUDA is byte-identical to an un-instrumented build.

    Args:
        primfunc: A built kernel whose body may contain ``trace.*`` markers.

    Returns:
        A ``PrimFunc`` with every marker replaced by a no-op; otherwise unchanged.
    """
    return _strip_markers(primfunc)


def lookup_meta(kernel) -> dict:
    """Resolve a compiled kernel back to its host decode maps.

    Reads the ``tl.trace_meta_id`` attr that ``lower`` stamped onto the
    transformed ``PrimFunc`` (exposed as ``kernel.prim_func``) and looks the maps
    up in ``_META_REGISTRY``.

    Args:
        kernel: A compiled kernel from a vanilla ``@tilelang.jit`` over a builder
            that returned ``lower``'s result.

    Returns:
        The host-map dict (``id_to_name`` / ``group_id_to_name`` /
        ``group_id_to_lead`` / ``lane_id_to_name`` / ``flows`` /
        ``max_events`` / ``num_groups`` / ``num_cta``).

    Raises:
        ValueError: If the kernel's ``prim_func`` carries no ``tl.trace_meta_id``
            attr (it was not produced by ``lower``), or the id is unknown.
    """
    prim_func = getattr(kernel, "prim_func", None)
    attrs = getattr(prim_func, "attrs", None)
    meta_id = attrs.get(_META_ATTR) if attrs is not None else None
    if meta_id is None:
        raise ValueError(
            "kernel was not produced by trace.lower (no tl.trace_meta_id attr on "
            "kernel.prim_func); cannot resolve host decode maps")
    meta_id = int(meta_id)
    if meta_id not in _META_REGISTRY:
        raise ValueError(f"unknown trace meta id {meta_id}; host maps not registered")
    return _META_REGISTRY[meta_id]
