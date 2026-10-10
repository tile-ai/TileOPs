"""Build a bytes-oracle case for each workload row of an op from its manifest entry alone.

Each row is instantiated, the op constructed from it and its call checked on meta tensors.
An implemented entry's op is its class; a spec-only entry's is a class carrying only what its
signature generates, since the recount needs no implementation.
In parallel the oracle counts the traffic the checked call implies -- one read per input it
binds, one write per output, both for a written input -- and the caller requires the two to
be equal. Empty experts read no weights, and a masked slab moves only its valid rows. The
`roofline` block is never read.

Shared with the formula, and nothing beyond it: the minimum-traffic definition and the
signature the call is checked against.
"""

from __future__ import annotations

import dataclasses

import torch

from tileops.manifest import load_adts, load_manifest
from tileops.manifest.plan import entry_plan
from tileops.manifest.registry import op_class
from tileops.manifest.workload import instantiate
from tileops.ops._signature_codegen import install
from tileops.ops.op_base import Op

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
        self.dispatch_kernel(None)

    body = {"__init__": construct, "default_kernel_map": property(lambda self: {})}
    body["forward"] = body["_eager_forward"] = lambda self, *args: None
    cls = type(f"Signature{op_name}", (Op,), body)
    if not install(cls, entry):
        raise ValueError(f"{op_name}: the signature does not generate")
    return cls


# The per-expert weights, and the tensors with one row per slab row, of the grouped MoE ops.
_EXPERT_WEIGHTS = {
    "MoEGroupedGemmFwdOp": ("b",),
    "MoEGroupedGemmFP8FwdOp": ("b", "b_scale"),
    "MoEExpertMLPFwdOp": ("w_gate_up", "w_down"),
}
_SLAB_ROWS = {
    "MoEGroupedGemmFwdOp": ("a", "output"),
    "MoEGroupedGemmFP8FwdOp": ("a", "a_scale", "output"),
    "MoEExpertMLPFwdOp": ("expert_input", "output"),
}


def _unused_expert_weights(op_name: str, call) -> int:
    weights = _EXPERT_WEIGHTS.get(op_name)
    if weights is None:
        return 0
    layout, experts = call.ix["layout"], call.ix["E"]
    metadata = call.values("layout_metadata")
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
    return sum(call.bytes(name) // experts * unused for name in weights)


def _unused_slab_rows(op_name: str, call) -> dict[str, int]:
    """Bytes of each row-shaped tensor in a masked slab's rows past its expert's count."""
    if op_name not in _SLAB_ROWS:
        return {}
    layout = call.ix["layout"]
    if layout.kind != "masked":
        return {}
    metadata = call.values("layout_metadata")
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
            metadata = {n: torch.tensor(call.values(n)) for n in checked.metadata}
            # The formula prices the op's last completed call; this one is that call.
            op._signature_call = dataclasses.replace(checked, metadata=metadata)
            unused = _unused_slab_rows(op_name, call)
            reads = sum((call.bytes(t) - unused.get(t, 0)) * r for t, r, _ in checked.traffic)
            reads -= _unused_expert_weights(op_name, call)
            writes = sum((call.bytes(t) - unused.get(t, 0)) * w for t, _, w in checked.traffic)
            yield row["label"], "-".join(case.values()), op, reads + writes, reads
