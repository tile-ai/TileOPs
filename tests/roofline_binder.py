"""Build a bytes-oracle case for each workload row of an op from its manifest entry alone.

Each row is instantiated, the op constructed from it and its call checked on meta tensors.
An implemented entry's op is its class; a spec-only entry's is a class carrying only what its
signature generates, since the recount needs no implementation.
In parallel the oracle counts the traffic the checked call implies -- one read per input it
binds, one write per output, both for a written input -- and the caller requires the two to
be equal. Empty experts and empty groups read no weights, a masked slab moves only its valid
rows, and attention reads only the query rows that see some key and the key and value rows
some query sees. The `roofline` block is
never read.

Shared with the formula, and nothing beyond it: the minimum-traffic definition and the
signature the call is checked against.
"""

from __future__ import annotations

import dataclasses
from math import prod

import torch

from tileops._ops._signature_codegen import install
from tileops._ops.op_base import Op
from tileops.manifest import load_adts, load_manifest
from tileops.manifest._plan import entry_plan
from tileops.manifest._registry import op_class
from tileops.manifest._workload import instantiate

__all__ = ["manifest_cases", "signature_class"]

# Keyed by the entry object: a caller may pass an edited copy of a manifest entry. The entry is
# held so its id is not reused.
_SIGNATURE_CLASSES: dict[tuple[str, int], tuple[dict, type]] = {}


def signature_class(op_name: str, entry: dict) -> type:
    """An `Op` subclass with *entry*'s generated methods and no kernel."""
    key = (op_name, id(entry))
    if key not in _SIGNATURE_CLASSES:
        _SIGNATURE_CLASSES[key] = (entry, _build_signature_class(op_name, entry))
    return _SIGNATURE_CLASSES[key][1]


def _build_signature_class(op_name: str, entry: dict) -> type:
    def construct(self, **params):
        vars(self).update(params)
        Op.__init__(self)

    body = {"__init__": construct}
    body["forward"] = lambda self, *args: None
    cls = type(f"Signature{op_name}", (Op,), body)
    if not install(cls, entry):
        raise ValueError(f"{op_name}: the signature does not generate")
    return cls


# The per-expert weights, and the tensors with one row per slab row, of the grouped MoE ops.
_EXPERT_WEIGHTS = {
    "MoEGroupedGEMMFwdOp": ("b", "bias"),
    "MoEGroupedGEMMFP8FwdOp": ("b", "b_scale"),
    "MoEExpertMLPFwdOp": ("w_gate_up", "w_down", "b_gate_up", "b_down"),
}
_SLAB_ROWS = {
    "MoEGroupedGEMMFwdOp": ("a", "output"),
    "MoEGroupedGEMMFP8FwdOp": ("a", "a_scale", "output"),
    "MoEExpertMLPFwdOp": ("expert_input", "output"),
}


def _unused_expert_weights(op_name: str, call) -> int:
    weights = _EXPERT_WEIGHTS.get(op_name)
    if weights is None:
        return 0
    layout, experts = call.indices["layout"], call.indices["E"]
    metadata = call.metadata_values("layout_metadata")
    if layout.kind == "masked":
        unused = sum(count == 0 for count in metadata)
    elif layout.metadata_kind == "per_row":
        unused = len(set(range(experts)) - set(metadata))
    else:
        # Reconstruct each expert's physical start independently of the formula.
        start, unused = 0, 0
        for end in metadata:
            unused += end == start
            start = end + (-end % layout.alignment)
    # An expert's optional bias is read with its weights, where the call passes one.
    return sum(call.bytes(name) // experts * unused for name in weights if call.present(name))


def _unused_group_matrices(op_name: str, call) -> int:
    """Bytes of the per-group matrices of ``b`` that a group with no row leaves unread."""
    if op_name != "GroupedGEMMFwdOp" or call.indices["trans_a"]:
        return 0
    sizes = call.metadata_values("batch_sizes")
    return call.bytes("b") // len(sizes) * sum(n == 0 for n in sizes)


def _dense_requests(call) -> list[tuple[int, int]]:
    return [(call.indices["S_q"], call.indices["S_kv"])] * call.indices["B"]


# The attention ops whose window or mask can leave a query row seeing no key, or a key row no
# query sees, with the per-request query and key lengths of a call.
_KV_REQUESTS = {
    "GQADenseFwdOp": _dense_requests,
    "GQADenseWithLSEFwdOp": _dense_requests,
    "GQAVarlenFwdOp": lambda call: list(
        zip(_lengths(call, "cu_seqlens_q"), _lengths(call, "cu_seqlens_kv"), strict=True)
    ),
}


def _lengths(call, name: str) -> list[int]:
    bounds = call.metadata_values(name)
    return [end - start for start, end in zip(bounds, bounds[1:], strict=False)]


def _unseen_attention_rows(op_name: str, call) -> dict[str, int]:
    """Bytes of the ``q`` rows that see no key, the ``k`` and ``v`` rows no query of their
    request sees, the RoPE-table positions no request reads and the per-request FP8 scales
    of requests that see nothing.

    Each query marks the keys it sees: query ``i`` sits at key ``i + kv - q``, sees the
    keys at most ``left`` before it, and, when not causal, at most ``right`` after it. The
    RoPE tables are indexed by request-relative position: a seen key is rotated at its
    position, and a query that sees some key sits at a position its own keys cover.
    """
    requests = _KV_REQUESTS.get(op_name)
    if requests is None:
        return {}
    causal = call.indices["is_causal"]
    left, right = call.indices["window_size_left"], call.indices["window_size_right"]
    seen_keys = seen_queries = active = 0
    positions = bytearray(call.tensors["rope_cos"][0][0] if call.present("rope_cos") else 0)
    for q, kv in requests(call):
        seen = bytearray(kv)
        for i in range(q):
            position = i + kv - q
            first = max(position - left, 0) if left >= 0 else 0
            if causal:
                last = min(position, kv - 1)
            else:
                last = min(position + right, kv - 1) if right >= 0 else kv - 1
            if last >= first:
                seen[first : last + 1] = bytes([1]) * (last + 1 - first)
                seen_queries += 1
                if positions:
                    positions[first : last + 1] = bytes([1]) * (last + 1 - first)
        seen_keys += sum(seen)
        active += any(seen)
    unseen = {}
    for names, seen_rows in ((("q",), seen_queries), (("k", "v"), seen_keys)):
        stored = prod(call.tensors[names[0]][0][:-2])
        for t in names:
            unseen[t] = call.bytes(t) - (call.bytes(t) // stored * seen_rows if stored else 0)
    if positions:
        for t in ("rope_cos", "rope_sin"):
            unseen[t] = call.bytes(t) // len(positions) * (len(positions) - sum(positions))
    requests_count = len(requests(call))
    for t in ("q_scale", "k_scale", "v_scale"):
        if call.present(t):
            unseen[t] = call.bytes(t) // requests_count * (requests_count - active)
    return unseen


def _nsa_compressed_unread(op_name: str, call) -> dict[str, int]:
    """Bytes of the ``q`` rows of tokens that close no chunk, and of the ``k_cmp`` and
    ``v_cmp`` chunks no token closes: token ``t`` of a request scores the chunks that end at
    or before it, chunks ``0 .. (t + 1) // block_size - 1`` of its request."""
    if op_name != "NSACompressedVarlenFwdOp":
        return {}
    block = call.indices["block_size"]
    idle_tokens = closed_chunks = 0
    for n in _lengths(call, "cu_seqlens"):
        scored = set()
        for t in range(n):
            chunks = range((t + 1) // block)
            idle_tokens += not chunks
            scored.update(chunks)
        closed_chunks += len(scored)
    unread = {}
    for names, read in (
        (("q",), call.tensors["q"][0][0] - idle_tokens),
        (("k_cmp", "v_cmp"), closed_chunks),
    ):
        stored = call.tensors[names[0]][0][0]
        for t in names:
            unread[t] = call.bytes(t) - (call.bytes(t) // stored * read if stored else 0)
    return unread


def _lightning_indexer_unread(op_name: str, call) -> dict[str, int]:
    """Bytes of the keys no query's window ``[starts[s], ends[s])`` covers, and of the query
    rows whose window is empty: ``index_k`` and ``index_k_scale`` per key of each batch row,
    ``index_q`` per query of each batch row, ``weights`` per query."""
    if op_name != "FP8LightningIndexerFwdOp":
        return {}
    starts, ends = call.metadata_values("starts"), call.metadata_values("ends")
    keys = call.indices["S_kv"]
    covered = bytearray(keys)
    for s, e in zip(starts, ends, strict=True):
        if e > s:
            covered[s:e] = bytes([1]) * (e - s)
    idle = sum(e <= s for s, e in zip(starts, ends, strict=True))
    queries = len(starts)
    unread = {}
    if queries:
        unread["index_q"] = call.bytes("index_q") // queries * idle
        unread["weights"] = call.bytes("weights") // queries * idle
    for t in ("index_k", "index_k_scale"):
        if call.present(t) and keys:
            unread[t] = call.bytes(t) // keys * (keys - sum(covered))
    return unread


def _unused_slab_rows(op_name: str, call) -> dict[str, int]:
    """Bytes of each row-shaped tensor in a masked slab's rows past its expert's count."""
    if op_name not in _SLAB_ROWS:
        return {}
    layout = call.indices["layout"]
    if layout.kind != "masked":
        return {}
    metadata = call.metadata_values("layout_metadata")
    padding = sum(layout.max_m - count for count in metadata)
    slab_rows = len(metadata) * layout.max_m
    return {name: call.bytes(name) // slab_rows * padding for name in _SLAB_ROWS[op_name]}


def manifest_cases(op_name: str):
    """Yield ``(label, dtype case, op, oracle bytes, oracle read bytes)`` per row and dtype case."""
    entry = load_manifest()[op_name]
    plan = entry_plan(op_name, entry, load_adts())
    cls = (
        op_class(op_name, entry)
        if entry["status"] == "implemented"
        else signature_class(op_name, entry)
    )
    for row in entry["workloads"]:
        for case in row.get("dtype_cases") or [{}]:
            call = instantiate(plan, row, case)
            tensors = call.materialize("meta")
            op = cls(**call.arguments(tensors))
            checked = type(op)._signature.check(op, {t: tensors[t] for t in plan.sig.inputs})
            # Meta tensors hold no values; the metadata a formula reads carries the row's own.
            metadata = {n: torch.tensor(call.metadata_values(n)) for n in checked.metadata}
            # The formula prices the op's last completed call; this one is that call.
            op._last_call = dataclasses.replace(checked, metadata=metadata)
            unused = (
                _unused_slab_rows(op_name, call)
                | _unseen_attention_rows(op_name, call)
                | _nsa_compressed_unread(op_name, call)
                | _lightning_indexer_unread(op_name, call)
            )
            reads = sum((call.bytes(t) - unused.get(t, 0)) * r for t, r, _ in checked.traffic)
            reads -= _unused_expert_weights(op_name, call) + _unused_group_matrices(op_name, call)
            writes = sum((call.bytes(t) - unused.get(t, 0)) * w for t, _, w in checked.traffic)
            yield row["label"], "-".join(case.values()), op, reads + writes, reads
