"""Derived facts of one manifest entry, computed once.

A check needs to know things the manifest does not state outright: which
tensors ``forward()`` receives, which columns a ``dtype_combos`` row spans,
which names an optional input binds. Each of those was re-derived at every
consumer, so a field that changed one of them — ``resources.workspaces`` was
the first — had to be chased through every consumer by hand, and a missed one
failed silently.

Every such fact is computed here and nowhere else. A consumer asks for the
fact; it does not decide what the fact is.

Parsing accumulates rather than aborts: a field that cannot be read yields
nothing for that fact while the rest of the entry is still read, because the
validator reports several schema problems per entry and must keep doing so.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SAME_AS_RE = re.compile(r"^\s*same_as\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*$")

#: Marks an entry of a signature's ``inputs`` that came from a workspace. A
#: caller may hand over a signature the workspaces were already merged into,
#: and the facts must still tell the two apart.
WORKSPACE_ATTR = "__workspace__"


class Section(Enum):
    """A mapping in the manifest whose keys are a closed set."""

    ENTRY = "entry"
    SIGNATURE = "signature"
    COMPOSITION = "composition"
    STAGE = "stage"
    VARIANT = "variant"
    RESOURCES = "resources"
    WORKSPACE = "workspace"
    ROOFLINE_COMPOSITION = "roofline.composition"


@dataclass(frozen=True)
class TensorArg:
    """One tensor of the call, with its dtype already parsed."""

    name: str
    dtype: str
    optional: bool = False
    workspace: bool = False
    #: The op writes this input in place and the manifest says so.
    mutated: bool = False
    shape: str | None = None

    @property
    def same_as(self) -> str | None:
        """The tensor this one's dtype follows, if it is declared that way."""
        match = SAME_AS_RE.match(self.dtype.strip())
        return match.group(1) if match else None


_SHAPE_DECL_RE = re.compile(r"^\s*\[([^\]]*)\]\s*$")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _shape_parts(shape: object) -> tuple[str, ...] | None:
    """Dimension names of a ``[a, b, c]`` declaration, or None.

    None where the declaration cannot be bound as mock dimension names: no
    shape, an empty one, or one carrying arithmetic or literals such as
    ``[4, d]``. A consumer distinguishes "declares no bindable shape" from
    "declares an empty shape", so the two must not collapse.
    """
    if not isinstance(shape, str):
        return None
    match = _SHAPE_DECL_RE.match(shape)
    if match is None:
        return None
    parts = tuple(part.strip() for part in match.group(1).split(",") if part.strip())
    if not parts or not all(_IDENT_RE.fullmatch(p) for p in parts):
        return None
    return parts


@dataclass(frozen=True)
class Facts:
    """Every derived fact of one entry. Consumers read; they do not re-derive."""

    name: str
    #: The declared status, or None where the entry did not state one as a string.
    status: str | None
    # inputs followed by workspaces, in declaration order: what forward() takes.
    call_tensor_args: tuple[TensorArg, ...] = ()
    # caller-visible inputs only: what a dtype_combos row and the reference
    # API are written against.
    value_inputs: tuple[TensorArg, ...] = ()
    outputs: tuple[TensorArg, ...] = ()
    combos: tuple[Mapping[str, str], ...] = ()
    #: Declaration order — the diagnostics print them that way.
    stage_names: tuple[str, ...] = ()
    #: Keys the entry carries that the parser does not read, per section.
    unknown_keys: Mapping[Section, tuple[Any, ...]] = field(default_factory=dict)
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    roofline: Mapping[str, Any] = field(default_factory=dict)

    # -- status and severity ----------------------------------------------

    @property
    def spec_only(self) -> bool:
        """Whether checks that need an implementation should stand down.

        ``status is None`` means the entry did not state one as a string.
        That counts as spec-only: schema reports the missing status, and a
        level run on its own must not go on to probe code that is not there.
        A status that is a string but not a known one is left alone — schema
        reports it, and nothing here second-guesses which branch it meant.
        """
        return self.status is None or self.status == "spec-only"

    @property
    def bench_manifest_driven(self) -> bool:
        """Whether the benchmark contract is a hard error rather than a warning."""
        return bool(self.source.get("bench_manifest_driven", False))

    # -- params ------------------------------------------------------------

    @property
    def mutated_input_names(self) -> frozenset[str]:
        """Caller inputs the manifest says the op writes in place.

        Workspaces are out by construction: every call writes one, so marking
        it would state nothing the reader did not already know.
        """
        return frozenset(a.name for a in self.value_inputs if a.mutated)

    # -- source -----------------------------------------------------------

    @property
    def source_paths(self) -> Mapping[str, str]:
        """The four declared paths, each as written inside the distribution."""
        return {
            key: value
            for key in ("kernel", "op", "test", "bench")
            if isinstance(value := self.source.get(key), str)
        }

    @property
    def kernel_map(self) -> Mapping[str, str]:
        """The op's dispatch table as the entry declares it, ``key -> Kernel``."""
        declared = self.source.get("kernel_map")
        if not isinstance(declared, dict):
            return {}
        return {k: v for k, v in declared.items() if isinstance(k, str) and isinstance(v, str)}

    @property
    def tensor_param_names(self) -> frozenset[str]:
        """Params typed as a tensor: where a caller-supplied output buffer sits.

        The op writes it and the return aliases it, so it is a param rather
        than an input.
        """
        return frozenset(
            name
            for name, attrs in self.params.items()
            if isinstance(attrs, dict) and "tensor" in str(attrs.get("type", "")).lower()
        )

    # -- names -------------------------------------------------------------

    @property
    def call_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.call_tensor_args)

    @property
    def optional_names(self) -> frozenset[str]:
        return frozenset(a.name for a in self.call_tensor_args if a.optional)

    @property
    def combo_columns(self) -> tuple[str, ...]:
        """Columns a ``dtype_combos`` row spans: caller inputs, workspaces out.

        A row states the dtype combinations a caller may pass. A workspace's
        dtype is execution strategy, so requiring a column for it would put
        strategy into the contract callers write against.
        """
        return tuple(a.name for a in self.value_inputs)

    @property
    def required_combo_columns(self) -> tuple[str, ...]:
        """Combo columns that every row must carry: optional inputs have none."""
        return tuple(a.name for a in self.value_inputs if not a.optional)

    # -- dtype -------------------------------------------------------------

    @property
    def same_as_map(self) -> Mapping[str, str]:
        """Tensor -> the tensor its dtype follows, over the call and the outputs.

        Only a bare ``same_as(ref)`` names one. A union that merely mentions it
        states a choice the caller makes, and reading that as an edge would
        demand a rejection the op never makes.
        """
        return {
            a.name: a.same_as
            for a in (*self.call_tensor_args, *self.outputs)
            if a.same_as is not None
        }

    @property
    def call_same_as_map(self) -> Mapping[str, str]:
        """Call argument -> the tensor its dtype follows.

        The negative dtype probes substitute an out-of-union dtype on one
        tensor and expect the ops that follow it to be rejected, so they need
        the edges among call arguments and nothing else.
        """
        return {a.name: a.same_as for a in self.call_tensor_args if a.same_as is not None}

    @property
    def declared_output_shapes(self) -> Mapping[str, tuple[str, ...]]:
        """Outputs whose shape the entry writes out, as dimension names.

        Absent rather than empty where the declaration cannot be bound as mock
        dimension names, because a consumer runs its check when the shape is
        stated and skips it when it is not.
        """
        out: dict[str, tuple[str, ...]] = {}
        for arg in self.outputs:
            parts = _shape_parts(arg.shape)
            if parts is not None:
                out[arg.name] = parts
        return out

    def arg(self, name: str) -> TensorArg | None:
        for a in (*self.call_tensor_args, *self.outputs):
            if a.name == name:
                return a
        return None


def _tensor_args(tensors: object, *, workspace: bool = False) -> tuple[TensorArg, ...]:
    if not isinstance(tensors, dict):
        return ()
    args = []
    for name, attrs in tensors.items():
        if not isinstance(name, str) or not isinstance(attrs, dict):
            continue
        dtype = attrs.get("dtype")
        args.append(
            TensorArg(
                name=name,
                dtype=dtype if isinstance(dtype, str) else "",
                optional=attrs.get("optional") is True,
                workspace=workspace or attrs.get(WORKSPACE_ATTR) is True,
                mutated=attrs.get("mutated") is True,
                shape=attrs.get("shape") if isinstance(attrs.get("shape"), str) else None,
            )
        )
    return tuple(args)


def _workspace_args(entry: Mapping[str, Any]) -> tuple[TensorArg, ...]:
    resources = entry.get("resources")
    workspaces = (resources or {}).get("workspaces") if isinstance(resources, dict) else None
    if not isinstance(workspaces, list):
        return ()
    args = []
    for ws in workspaces:
        if not isinstance(ws, dict) or not isinstance(ws.get("name"), str):
            continue
        dtype = ws.get("dtype")
        args.append(
            TensorArg(
                name=ws["name"],
                dtype=dtype if isinstance(dtype, str) else "",
                optional=ws.get("optional") is True,
                workspace=True,
            )
        )
    return tuple(args)


#: Keys each closed section accepts. Declared where the parser reads them so
#: the diagnostic and the parser cannot disagree about what is valid.
_ENTRY_KEYS = (
    "family",
    "status",
    "signature",
    "workloads",
    "roofline",
    "source",
    "ref_api",
    "torch_compile_fullgraph",
    "composition",
    "resources",
)
_SIGNATURE_KEYS = ("inputs", "outputs", "params", "shape_rules", "dtype_combos", "static_dims")
_COMPOSITION_KEYS = ("kind", "stages")
_STAGE_KEYS = ("name", "op", "kernel", "variants", "optional")
_VARIANT_KEYS = ("name", "condition", "stages")
_RESOURCE_KEYS = ("workspaces",)
_WORKSPACE_KEYS = ("name", "dtype", "owner", "kind", "optional", "note")
_ROOFLINE_COMPOSITION_KEYS = ("stage", "source", "formula", "optional")

#: The accepted key set of each closed section, for the diagnostic that prints
#: it. One declaration, read both by the parser and by the message.
SECTION_KEYS: Mapping[Section, tuple[str, ...]] = {
    Section.ENTRY: _ENTRY_KEYS,
    Section.SIGNATURE: _SIGNATURE_KEYS,
    Section.COMPOSITION: _COMPOSITION_KEYS,
    Section.STAGE: _STAGE_KEYS,
    Section.VARIANT: _VARIANT_KEYS,
    Section.RESOURCES: _RESOURCE_KEYS,
    Section.WORKSPACE: _WORKSPACE_KEYS,
    Section.ROOFLINE_COMPOSITION: _ROOFLINE_COMPOSITION_KEYS,
}


def unknown_keys_of(section: Section, raw: object) -> tuple[Any, ...]:
    """Keys *raw* carries that the parser does not read for this section."""
    if not isinstance(raw, dict):
        return ()
    accepted = set(SECTION_KEYS[section])
    return tuple(sorted((k for k in raw if k not in accepted), key=repr))


def build(name: str, entry: Mapping[str, Any]) -> Facts:
    """Read one entry into its facts, accumulating what cannot be read."""
    sig = entry.get("signature")
    sig = sig if isinstance(sig, dict) else {}
    unknown = {
        Section.ENTRY: unknown_keys_of(Section.ENTRY, entry),
        Section.SIGNATURE: unknown_keys_of(Section.SIGNATURE, sig),
    }

    merged_inputs = _tensor_args(sig.get("inputs"))
    # A signature handed in already merged carries its workspaces inside
    # ``inputs``; the value contract is the rest.
    value_inputs = tuple(a for a in merged_inputs if not a.workspace)
    premerged = tuple(a for a in merged_inputs if a.workspace)
    outputs = _tensor_args(sig.get("outputs"))

    # A workspace is declared apart from the inputs but passed like one, so the
    # call carries both; only the inputs are the caller's value contract.
    workspaces = (*premerged, *_workspace_args(entry))
    declared = {a.name for a in value_inputs}
    seen: set[str] = set()
    ordered_ws = []
    for w in workspaces:
        if w.name in declared or w.name in seen:
            continue
        seen.add(w.name)
        ordered_ws.append(w)
    call = (*value_inputs, *ordered_ws)

    raw_combos = sig.get("dtype_combos")
    combos: tuple[Mapping[str, str], ...] = ()
    if isinstance(raw_combos, list):
        combos = tuple(c for c in raw_combos if isinstance(c, dict))

    composition = entry.get("composition")
    stage_names: tuple[str, ...] = ()
    if isinstance(composition, dict):
        stages = composition.get("stages")
        if isinstance(stages, list):
            stage_names = tuple(
                st["name"]
                for st in stages
                if isinstance(st, dict) and isinstance(st.get("name"), str)
            )

    raw_params = sig.get("params")
    raw_source = entry.get("source")
    raw_roofline = entry.get("roofline")
    status = entry.get("status")
    return Facts(
        name=name,
        status=status if isinstance(status, str) else None,
        call_tensor_args=call,
        value_inputs=value_inputs,
        outputs=outputs,
        combos=combos,
        stage_names=stage_names,
        unknown_keys=unknown,
        params=raw_params if isinstance(raw_params, dict) else {},
        source=raw_source if isinstance(raw_source, dict) else {},
        roofline=raw_roofline if isinstance(raw_roofline, dict) else {},
    )
