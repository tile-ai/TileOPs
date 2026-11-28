import copy
import dataclasses
import functools
import inspect
import math
import threading
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
)

import torch
from torch._subclasses.fake_tensor import FakeTensor

from tileops._kernels.build_target import building_on, target_of
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops._signature_codegen import (
    PARAM_NAMES_ATTRIBUTE,
    CheckError,
    ResultCheck,
    SignatureCall,
    _snapshot,
    in_tree,
    install_param_names,
    leaves,
    maybe_install_signature,
    op_entry,
    param_state,
    tensorless_device,
)
from tileops._ops.compile_boundary import instance_key, register_instance
from tileops._utils.device import current_device, device_context
from tileops.backend import (
    BUILTIN,
    OpNotAvailableError,
    OpNotServedError,
    Target,
    TensorSpec,
    registered_targets,
)
from tileops.backend._dispatch import registered_kernel_builder, select_target
from tileops.backend._registry import REGISTERED_KERNEL_TYPES, ensure_loaded


class _Unresolved:
    """The type of :data:`_UNRESOLVED`, so a traceback says what it is."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<not resolved yet>"


# ``Op._target_builder`` before the first call. Distinct from ``None``, the decided answer
# "run the in-tree implementation".
_UNRESOLVED = _Unresolved()


class _OpenCalls(threading.local):
    """The calls in progress on this thread, innermost last: each one's frame."""

    def __init__(self) -> None:
        """Start each thread with no call in progress."""
        self.stack: list = []


_OPEN_CALLS = _OpenCalls()

# ``Op._execute``'s *key* when its caller has not observed the call.
_UNOBSERVED = object()


def _address(t: object) -> "int | None":
    """The address *t* starts at, or None for an absent value, a scalar, or a tensor whose
    storage has no address: a fake one, or a wrapper refusing the read."""
    if not isinstance(t, torch.Tensor) or isinstance(t, FakeTensor):
        return None
    try:
        return t.data_ptr()
    except RuntimeError:
        return None


def _slot_spec(t: object) -> object:
    """What a kernel builder is told of the input *t*: a ``TensorSpec`` for a tensor, a tuple of
    them for a tensor sequence, ``None`` for an absent input, and its type for a call-time
    scalar."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        return tuple(map(TensorSpec.of, t))
    return TensorSpec.of(t) if isinstance(t, torch.Tensor) else type(t)


def _element_span(t: torch.Tensor) -> int:
    """The elements from the first element of *t*, which holds one, to its last, holes of a
    strided layout included."""
    return sum((n - 1) * s for n, s in zip(t.shape, t.stride(), strict=True)) + 1


def _overlaps_itself(t: torch.Tensor) -> bool:
    """Whether two elements of *t*, which holds one, share an address.

    torch decides most layouts. One it cannot is disjoint when, its dimensions of more than
    one element ordered by stride, each stride passes the farthest element the smaller ones
    reach, and overlaps when it holds more elements than the addresses it spans; any other
    is the caller's obligation, since deciding it reads every offset.
    """
    overlap = torch._debug_has_internal_overlap(t)
    if overlap != 2:
        return overlap == 1
    reach = 0
    for stride, size in sorted((s, n) for n, s in zip(t.shape, t.stride(), strict=True) if n > 1):
        if stride <= reach:
            return t.numel() > _element_span(t)
        reach += (size - 1) * stride
    return False


def _byte_layout(t: torch.Tensor) -> tuple:
    """``(dims, element bytes)`` of *t*: each dimension of more than one element as
    ``(elements, stride in bytes)``."""
    size = t.element_size()
    dims = tuple((n, s * size) for n, s in zip(t.shape, t.stride(), strict=True) if n > 1)
    return dims, size


def _apart(a: int, first: tuple, b: int, second: tuple) -> bool:
    """Whether tensors of byte layouts *first* and *second* (:func:`_byte_layout`) starting
    at addresses *a* and *b* provably share no byte though their spans meet.

    They do where some period ``p``, one of their strides, divides every stride of either at
    least ``p``: each tensor's bytes then lie, modulo ``p``, on one arc from its start's
    residue, as long as its strides below ``p``, which leave no other trace, span; two arcs
    that do not meet hold no common byte. Column slices of one row-major buffer pass with the
    row as ``p``.
    """
    (dims_a, size_a), (dims_b, size_b) = first, second
    for p in sorted({s for _, s in (*dims_a, *dims_b) if s > 0}):
        arcs = []
        for dims, size in ((dims_a, size_a), (dims_b, size_b)):
            if any(s >= p and s % p for _, s in dims):
                break
            arcs.append(sum((n - 1) * s for n, s in dims if s < p) + size)
        else:
            gap = (b - a) % p
            if arcs[0] <= gap and arcs[1] <= p - gap:
                return True
    return False


def _overlap_checks(
    op_name: str, names: tuple, tensors: tuple, written: frozenset, overlaps: Mapping
) -> tuple:
    """The ``(position, other position, span, other span, layout, other layout, message)`` of
    each pair of *tensors*, in slots *names*, that must share no byte: one of the two is in
    *written*, neither names the other in its *overlaps* entry, and both hold elements on one
    device off the meta device. A span is the bytes from a tensor's first element to the end of
    its last; a layout is its :func:`_byte_layout`, which tells interleaved tensors apart.
    """
    if not written:
        return ()
    spans = [
        0
        if not isinstance(t, torch.Tensor) or t.is_meta or not t.numel()
        else _element_span(t) * t.element_size()
        for t in tensors
    ]
    checks = []
    for i, name in enumerate(names):
        if name not in written or not spans[i]:
            continue
        for j, other in enumerate(names):
            # A pair of written slots is checked once, from the earlier one.
            if j == i or not spans[j] or (other in written and j < i):
                continue
            if other in overlaps.get(name, ()) or name in overlaps.get(other, ()):
                continue
            if tensors[i].device == tensors[j].device:
                message = f"{op_name}: {name} must not overlap {other} in memory"
                layouts = (_byte_layout(tensors[i]), _byte_layout(tensors[j]))
                checks.append((i, j, spans[i], spans[j], *layouts, message))
    return tuple(checks)


def _aligned(
    op: "Op", plan: "_Plan", inputs: tuple, writes: "dict[str, torch.Tensor]", written: frozenset
) -> "tuple[tuple, dict, tuple]":
    """*inputs* and *writes* with each tensor at a position in ``plan.copied``, counting
    *inputs* then *writes*, replaced by an aligned copy, and the ``(copy, caller's tensor)``
    pairs to copy back.

    A copy holds the caller's values, so an element the implementation leaves unwritten keeps
    them. *written* names the inputs the implementation writes; every entry of *writes* is
    written. A copy hides no overlap: the call has been checked to write no tensor sharing a
    byte with another.
    """
    passed = leaves((*inputs, *writes.values()))
    back = plan.staging.get(written)
    if back is None:
        names = _leaf_names((*type(op)._signature.sig.inputs, *writes), (*inputs, *writes.values()))
        writing = written.union(writes) if writes else written
        back = plan.staging[written] = tuple(i for i in plan.copied if names[i] in writing)
    tensors = list(passed)
    for i in plan.copied:
        tensors[i] = passed[i].clone()
    count = len(leaves(inputs))
    if writes:
        writes = dict(zip(writes, tensors[count:], strict=True))
    back = tuple([(tensors[i], passed[i]) for i in back]) if back else ()
    return _regrouped(inputs, tensors[:count]), writes, back


def _leaf_names(names: tuple, slots: tuple) -> tuple:
    """The slot name of each of :func:`leaves` of *slots*, whose names are *names*."""
    return tuple(
        name
        for name, value in zip(names, slots, strict=True)
        for _ in (value if isinstance(value, (list, tuple)) else (value,))
    )


def _regrouped(slots: tuple, flat: list) -> tuple:
    """*slots* with their :func:`leaves` replaced, in order, by *flat*: a tensor sequence keeps
    its list or tuple type."""
    out, k = [], 0
    for value in slots:
        if isinstance(value, (list, tuple)):
            out.append(type(value)(flat[k : k + len(value)]))
            k += len(value)
        else:
            out.append(flat[k])
            k += 1
    return tuple(out)


def _restored(result: object, back: tuple) -> object:
    """*result* with each written copy in *back* replaced by the caller's tensor, in the
    sequence type *result* has."""
    callers = {id(copy): caller for copy, caller in back}
    if not isinstance(result, (tuple, list)):
        return callers.get(id(result), result)
    items = [callers.get(id(t), t) for t in result]
    kind = type(result)
    if kind in (tuple, list):
        return kind(items)
    # A named tuple takes its fields one by one; a structseq takes one sequence.
    return kind._make(items) if hasattr(kind, "_make") else kind(items)


class _Plan:
    """What one specialization of an op's calls decides, resolved on its first call.

    A specialization is everything the generated checks, the call specs and the plan read:
    each tensor's shape, strides, dtype and device, whether each call tensor starts on a
    16-byte boundary, which members the call-time arguments hold, the op's params, the
    execution arguments, and the current device of a call no tensor places. The
    plan holds the checked call's facts, the call tensors it copies to a 16-byte boundary, the
    pairs of call tensors that must share no byte, its effect branch, its result check, whether
    it writes no element, and each kernel request ``forward`` made, in request order. It holds
    no tensor a call passes, returns or completes.
    """

    __slots__ = (
        "call",
        "copied",
        "overlaps",
        "metadata",
        "contents",
        "effect",
        "empty",
        "result_check",
        "stages",
        "staging",
        "positional",
        "requests",
        "delegates",
        "filling",
        "scalar",
    )

    def __init__(
        self, signature, call, copied: tuple, overlaps: tuple, effect: tuple, empty: bool, stages
    ) -> None:
        """Hold what *signature*'s checks decided for *call*, without its metadata tensors;
        *copied* is the positions of the call tensors, inputs then writes, that do not start
        on a 16-byte boundary; *overlaps* is what ``_overlap_checks`` returns for the call;
        *stages* is the op's stage mapping of a call that collected nothing."""
        sig = signature.sig
        order = list(sig.inputs)
        # Its own copy of the indices: a param a call reads can be a list or an ADT, which
        # the op's later edits must not reach.
        self.call = dataclasses.replace(call, indices=copy.deepcopy(call.indices), metadata=None)
        # Each metadata tensor by name, with its position among the inputs.
        self.metadata = tuple((name, order.index(name)) for name in call.metadata)
        # The `validate_inputs` check of the call's tensor contents, or None when it has none.
        self.contents = signature.contents.get(call.branch_key)
        self.copied, self.effect, self.empty, self.stages = copied, effect, empty, stages
        self.overlaps = overlaps
        # The copied positions to copy back, by the inputs the implementation writes.
        self.staging: dict = {}
        self.result_check = ResultCheck(sig, call)
        self.positional = signature.positional
        # Each kernel request ``forward`` made, in request order.
        self.requests: list = []
        # Each ``(stage, sub-op)`` ``forward`` held under no identity, in request order.
        self.delegates: list = []
        # Held while a call extends either list, so calls missing together extend it once.
        # Reentrant: a sub-op built under it may resolve its own plan.
        self.filling = threading.RLock()
        # Whether a call-time argument holds a scalar member: its value may decide a call
        # spec and the plan holds no scalar value, so each request builds its spec.
        self.scalar = any(m in sig.inputs[t].scalars for t, m in (call.members or {}).items())


class _Request:
    """One kernel request a plan holds: its interface, the code of the function that built
    its call spec, the spec, and ``(entry table, entry)`` once the spec resolved."""

    __slots__ = ("interface", "code", "spec", "resolved")

    def __init__(self, interface: str, code: object, spec: object) -> None:
        """Hold a request whose spec has not resolved to an entry yet."""
        self.interface, self.code, self.spec, self.resolved = interface, code, spec, None


class _Frame:
    """One call in progress: its op, its plan, and the calls its sub-ops completed inside it.

    ``request_index`` is the position of ``forward``'s next kernel request in ``plan.requests``,
    and ``delegate_index`` that of its next request of a sub-op held under no identity in
    ``plan.delegates``.
    """

    __slots__ = ("op", "plan", "children", "request_index", "delegate_index")

    def __init__(self, op: "Op", plan: "_Plan | None") -> None:
        """Open a call of *op* under *plan*, before any sub-op call or kernel request."""
        self.op, self.plan, self.children, self.request_index = op, plan, [], 0
        self.delegate_index = 0


class _CallRecord:
    """A completed call: the facts its plan checked, the metadata tensors it passed, and the
    records of the calls its sub-ops completed inside it, by stage in completion order.

    It refers to no op and no plan, so nothing that happens later changes it. Its
    ``SignatureCall`` is built on the first read and the same object on every later one.
    """

    __slots__ = ("template", "metadata", "children", "empty_stages", "built")

    def __init__(self, template, metadata, children: tuple, empty_stages) -> None:
        """Record a call of *template*'s facts, with its *metadata* tensors and *children*,
        ``(stage, record)`` pairs, under the op's *empty_stages* mapping."""
        self.template, self.metadata, self.children = template, metadata, children
        self.empty_stages, self.built = empty_stages, None

    def materialize(self):
        """The completed call as a ``SignatureCall``, built on the first read."""
        built = self.built
        if built is None:
            if self.children:
                stages = {stage: [] for stage in self.empty_stages}
                for stage, record in self.children:
                    stages[stage].append(record.materialize())
                stages = {stage: tuple(done) for stage, done in stages.items()}
            else:
                stages = self.empty_stages
            # Copies: the template's indices and tensors are the plan's, which later records
            # share.
            template = self.template
            built = self.built = dataclasses.replace(
                template,
                indices=copy.deepcopy(template.indices),
                tensors=dict(template.tensors),
                metadata=self.metadata,
                stages=stages,
            )
        return built


class Op(ABC):
    """Base class of every TileOPs op.

    For each call it checks the arguments against the op's manifest signature, selects the
    target, gets or builds the kernels, checks and records the result, and undoes what a
    failed call selected. A class with a manifest entry is given, when it is defined, its
    signature check, output-shape inference, ``eval_roofline`` when the entry has a
    ``roofline``, and a compile boundary when the entry has a call-time tensor input and no
    composition. A class in ``tileops`` takes the manifest entry of its name; a class defined
    elsewhere states its entry as the class attribute ``spec``, a mapping with the fields of
    one manifest entry (``family``, ``signature``, and optionally ``roofline`` and
    ``workloads``), and subclasses ``Op`` itself, never an op of ``tileops``. A subclass
    assigns its manifest parameters, calls ``super().__init__(target=target)``, and writes its
    computation in ``forward``; it builds derived attributes and sub-ops after
    ``super().__init__``. It declares ``kernel_types``, ``interfaces``, ``delegate_types`` and
    ``injected_parameters``; it may override ``dtype``, ``roof_key()``,
    ``eval_roofline_read_bytes()`` and ``roofline_data_terms()``. Callers call the op as
    ``op(...)``, never ``forward``. Each member states its contract in its own docstring or
    comment; docs/design/ops-design-reference.md describes the call protocol and the kernel
    caching methods.

    A member whose name starts with an underscore belongs to the base class and its generated
    code; a subclass neither calls nor overrides it.
    """

    # Which set of kernels serves this instance: a target name, ``BUILTIN`` for the in-tree
    # implementation, or None to decide from the input device. Constructor-only: it settles
    # kernel identity, so it must not vary per call.
    target: Target = None
    # The bound builder: ``_UNRESOLVED`` before a call binds one, ``None`` for the in-tree
    # kernels, or a target's build_kernel.
    _target_builder: object = _UNRESOLVED
    # The target bound with it, for ``serving_target``.
    _serving_target: Target = None
    # Whether this instance has warned that a kernel of its target cannot be tuned.
    _tune_warned: bool = False
    # Whether this instance is in tuned mode. Only ``request_tune()`` sets it; every entry and
    # sub-op built while it is set is put in tuned mode too.
    _tune_requested: bool = False

    # The implementation installed under each key, in-tree or registered by a backend.
    # A composite that declares no kernel of its own keeps this empty class value.
    _installed_kernel_types: Mapping[str, type[Kernel]] = MappingProxyType({})
    # Each key mapped to the keys its implementation is preferred over, transitively.
    _preferred_over: Mapping[str, frozenset[str]] = MappingProxyType({})
    # The keys of each kernel interface's implementations, as installed.
    _keys_by_interface: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    # The binding caches, built empty by ``__init__`` and emptied by ``_reset_binding``: resolved
    # entries by ``(interface, call spec)``, built entries by interface and
    # ``(implementation, identity)``, and the kernels a target built by device and inputs.
    _entries_by_call: dict[tuple[str, Hashable], object]
    # The most call specs ``_entries_by_call`` holds; the oldest is evicted past it.
    _MAX_DISPATCHED_CALLS: ClassVar[int] = 256
    _built_entries: dict[str, dict[Hashable, object]]
    _target_kernels: dict[tuple, object]
    # Held sub-ops, ``{stage: {identity: op}}``, and which stage holds each sub-op, by its
    # ``id``. Built empty by ``__init__``; a failed call keeps them.
    _delegates: dict[str, dict[Hashable, "Op"]]
    _delegate_stages: dict[int, str]
    # The effect branch of each presence of the optional inputs, with the param values it
    # was looked up under, filled by the generated compile boundary. Built empty by
    # ``__init__``.
    _effect_branches: dict[tuple, tuple]
    # The plan of each specialization this instance has resolved, by its key
    # (``_observe``). Built empty by ``__init__``.
    _plans: dict
    # The most plans ``_plans`` holds; the oldest is evicted past it. Bounded because calls
    # whose shapes keep changing, such as decode steps over a growing cache, resolve one each.
    _MAX_PLANS: ClassVar[int] = 256
    # The generated signature plan; None for a class without a manifest entry.
    _signature: ClassVar[object] = None
    dtype: torch.dtype | None = None
    # Whether a call's tensor contents are checked against their `requires`; set by every op
    # whose signature declares `validate_inputs`.
    validate_inputs: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Install what the subclass's manifest entry generates, and the param names a
        backend's ``build_kernel`` is called with.

        A subclass without an entry gets neither. An entry's ``status`` does not change what
        its signature generates. A class given a compile boundary is called through it
        directly; any other class with an entry through :meth:`__call__`.

        Raises:
            TypeError: A class outside ``tileops`` subclasses an op class of ``tileops``
                other than ``Op``, or its ``spec`` is not a mapping.
            ValueError: A class outside ``tileops`` states a ``spec`` the manifest validator
                rejects.
        """
        super().__init_subclass__(**kwargs)
        if not in_tree(cls):
            for base in cls.__mro__[1:]:
                if base is not Op and issubclass(base, Op) and in_tree(base):
                    raise TypeError(
                        f"{cls.__name__} subclasses {base.__name__}, an op of tileops; an op "
                        f"defined outside tileops subclasses Op and states its own spec"
                    )
        entry = op_entry(cls)
        if maybe_install_signature(cls, entry):
            cls.__call__ = cls.__dict__.get("_call_boundary", Op.__call__)
        install_param_names(cls, entry)

    # The op's dispatch keys and the kernel class each names. An op with no kernel of its own
    # (a composite) declares none.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = MappingProxyType({})

    # The places this op calls a kernel, each named and mapped to its kernel interface. An
    # interface's implementations are the keys whose registered class inherits it, in-tree or
    # added through ``tileops.backend._registry.register_kernel_type``.
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = MappingProxyType({})

    # The ops this op holds as sub-ops, by stage name in stage order: the sub-op counterpart of
    # ``kernel_types``. Every sub-op is held through ``delegate_for``.
    delegate_types: ClassVar[Mapping[str, type["Op"]]] = MappingProxyType({})

    @property
    def last_call(self) -> SignatureCall:
        """The ``SignatureCall`` of this op's last completed call: its ``indices``, tensors and effects.

        The same object on every read until another call completes.

        Raises:
            RuntimeError: No call has completed yet.
        """
        record = getattr(self, "_last_call", None)
        if record is None:
            raise RuntimeError(f"{type(self).__name__}: no call has completed yet")
        return record.materialize()

    # Operators this op registers on the torch.compile boundary; non-empty exactly when the
    # op has one. The signature generates them for a class whose entry has a call-time tensor
    # input and no composition, and such an op must compile cold with ``fullgraph=True``.
    # Naming them lets a test assert the traced graph holds nothing else, which keeps the
    # graph the same when another target serves the op. A conditional in-place write
    # registers two.
    compile_op_names: ClassVar[tuple[str, ...]] = ()

    # The attributes the compile boundary's fake reads, in key order: the signature's params.
    # Assigning one changes `_instance_key`.
    _instance_key_names: ClassVar[tuple[str, ...]] = ()
    # The manifest entry the fake is generated from, as text.
    _instance_key_entry: ClassVar[str] = ""
    # Serializes the assignments that change an op's compile-boundary key.
    _instance_key_lock: ClassVar[threading.Lock] = threading.Lock()

    # Injected implementation objects ``__init__`` takes beyond ``signature.params`` and the
    # execution-policy parameters every op takes (docs/design/manifest.md § Signature).
    injected_parameters: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, target: Target = None) -> None:
        """Do the part of construction every op shares, once the subclass has assigned its
        manifest parameters.

        Loads the backend registry, checks the manifest parameters, installs the
        implementations and registers the instance for its compile boundary. It selects and
        builds no kernel and reads no device property: an op constructs wherever it is
        imported, and a device it cannot run on is refused at the first call.

        Args:
            target: Which set of kernels serves this op: a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the call's device.

        Raises:
            ValueError: A manifest parameter is outside its signature, or an implementation
                breaks its kernel interface's contract.
            TypeError: An implementation breaks its kernel interface's contract.
        """
        self.target = target
        # Before any traced region, which the first call may be inside.
        ensure_loaded()
        check = getattr(type(self), "_check_construction", None)
        if check is not None:
            self._construction_indices = check(self)
        self._install_kernel_types()
        self._instance_identity = register_instance(self)
        self._instance_key = instance_key(self._instance_identity, self._instance_key_state())
        # The binding caches and the sub-op tables, so no method asks whether they exist;
        # installing built the two entry caches.
        self._target_kernels = {}
        # Held while an entry is built; reentrant, since a build may resolve another entry.
        self._building = threading.RLock()
        self._delegates = {}
        self._delegate_stages = {}
        self._effect_branches = {}
        self._plans = {}

    def __setattr__(self, name: str, value: object) -> None:
        if name not in self._instance_key_names or "_instance_identity" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        # Keeps a concurrent assignment from publishing the key of a value it replaced.
        with self._instance_key_lock:
            object.__setattr__(self, name, value)
            self._instance_key = instance_key(self._instance_identity, self._instance_key_state())

    def _instance_key_state(self) -> tuple:
        """What the compile boundary's fake reads of this op, in `_instance_key_names` order."""
        return (
            self._instance_key_entry,
            *(getattr(self, n, None) for n in self._instance_key_names),
        )

    def _infer_output_shapes(self, *shapes: tuple[int, ...], dtypes=None) -> dict:
        """Infer output tensor shapes from input shapes, in ``signature.inputs`` order.

        Generated from the op's manifest entry.

        Raises:
            TypeError: The op has no manifest entry.
        """
        raise self._no_entry()

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last completed call.

        Generated from the op's manifest ``roofline`` (docs/design/roofline.md §4.4).

        Raises:
            NotImplementedError: The op's manifest entry has no ``roofline``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: its manifest entry has no roofline, so eval_roofline "
            f"has no formula"
        )

    def _no_entry(self) -> TypeError:
        """The error a call of an op without a manifest entry raises."""
        name = type(self).__name__
        if in_tree(type(self)):
            return TypeError(f"{name} has no manifest entry, so it cannot be called")
        return TypeError(
            f"{name} states no spec, so it cannot be called; an op defined outside tileops "
            f"states its manifest entry as the class attribute `spec`"
        )

    def eval_roofline_read_bytes(self) -> int | None:
        """The read half of ``eval_roofline()[1]``, for the NCU bytes audit: ``bytes`` minus
        the write half the signature settles (every output once, plus each written input that
        is not an output).

        Returns:
            The read half in bytes, or ``None`` from an override whose read half the
            signature cannot settle.
        """
        call = self.last_call
        write_bytes = sum(call.bytes(t) * w for t, _, w in call.traffic)
        return int(self.eval_roofline()[1]) - write_bytes

    def roofline_data_terms(self) -> "dict[str, int]":
        """The input values that decided this call's ``bytes``, such as a routed MoE's selected
        experts, which the benchmark records beside the reading. Empty unless the op's traffic
        follows its inputs' values.
        """
        return {}

    def roof_key(self) -> str:
        """GPU-profile key of the compute unit whose peak bounds ``eval_roofline()``'s FLOPs
        (docs/design/roofline.md §1.2).

        It names the unit of the optimal implementation, never the running kernel's. The
        default is fp32 CUDA cores; an op whose FLOPs are matmul contractions overrides it with
        ``tensor_core_roof`` of the contraction's input dtype from ``self.last_call``.
        """
        return "cuda_core.fp32"

    def _install_kernel_types(self) -> None:
        """Install the implementation of each key, from ``kernel_types`` or a backend.

        Resolving a kernel *class* needs no device, so installation probes none: a target
        that cannot run the op surfaces when a kernel is first selected, built or called. A
        composite that declares no kernel of its own installs nothing; its sub-ops install
        their own. Installing again drops every entry the previous installation built.

        Raises:
            ValueError: A backend registered a key this op already has, or what
                :meth:`_check_and_index_keys` raises.
            TypeError: What :meth:`_check_and_index_keys` raises.
        """
        # An entry built from a previous installation may come from a kernel type the new
        # one no longer selects, so installing again drops every in-tree entry.
        self._entries_by_call = {}
        self._built_entries = {}
        if not self.kernel_types:
            return
        name = type(self).__name__
        added = REGISTERED_KERNEL_TYPES.get(name, {})
        taken = sorted(set(added) & set(self.kernel_types))
        if taken:
            raise ValueError(f"implementations registered for {name} reuse keys it has: {taken}")
        registered = {**self.kernel_types, **added}
        self._installed_kernel_types = MappingProxyType(registered)
        preferred = {}
        for key in registered:
            seen: set[str] = set()
            pending = list(getattr(registered[key], "preferred_over", ()))
            while pending:
                above = pending.pop()
                if above not in seen and above in registered:
                    seen.add(above)
                    pending.extend(getattr(registered[above], "preferred_over", ()))
            preferred[key] = frozenset(seen)
        self._preferred_over = MappingProxyType(preferred)
        self._keys_by_interface = self._check_and_index_keys()

    def _check_and_index_keys(self) -> Mapping[str, tuple[str, ...]]:
        """Return each kernel interface's keys, each implementation checked against it.

        A key belongs to every interface its implementation inherits. The implementation
        must take the interface's ``forward`` arguments and state ``entry_for`` as a
        classmethod.

        Raises:
            TypeError: An implementation breaks its interface's contract, or declares
                ``supported_archs`` for more than one device type.
            ValueError: A key belongs to no interface, an interface has two general
                implementations, or a ``preferred_over`` names no other implementation of
                its interface, is stated by the general one, or is cyclic.
        """
        name = type(self).__name__
        interface_keys = {}
        for where, interface in self.interfaces.items():
            keys = tuple(
                k for k, cls in self._installed_kernel_types.items() if issubclass(cls, interface)
            )
            parameters = list(inspect.signature(interface.forward).parameters.values())[1:]
            # Each argument as the interface takes it: by position, or by name after `*`.
            arguments = [p.name for p in parameters if p.kind is not p.KEYWORD_ONLY]
            keywords = {p.name: p.name for p in parameters if p.kind is p.KEYWORD_ONLY}
            for key in keys:
                runs = self._installed_kernel_types[key]
                if not issubclass(runs, Kernel):
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__} does not implement "
                        f"{interface.__name__}; an implementation is a Kernel that inherits "
                        f"{interface.__name__} and is built through its classmethod "
                        f"entry_for(call)"
                    )
                if not isinstance(inspect.getattr_static(runs, "entry_for"), classmethod):
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__}.entry_for is not a classmethod"
                    )
                if runs.supported_archs is not None and len(runs.devices) != 1:
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__} declares supported_archs "
                        f"for devices {sorted(runs.devices)}; an architecture compares within "
                        f"one device type, so an implementation declaring them names one"
                    )
                try:
                    inspect.signature(runs.forward).bind(None, *arguments, **keywords)
                except TypeError as exc:
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__}.forward does not take "
                        f"{interface.__name__}'s arguments: {exc}"
                    ) from None
            general = [k for k in keys if self._installed_kernel_types[k].general]
            if len(general) > 1:
                raise ValueError(
                    f"{name}.{where} has more than one general implementation: {general}"
                )
            for key in keys:
                preferred = self._installed_kernel_types[key].preferred_over
                if not preferred <= set(keys) - {key} or (preferred and key in general):
                    raise ValueError(
                        f"{name}.{where} {key!r} is preferred over {sorted(preferred)}; it names "
                        f"other implementations of its interface, and the general one names none"
                    )
                if key in self._preferred_over[key]:
                    raise ValueError(f"{name}.{where} preferences form a cycle through {key!r}")
            interface_keys[where] = keys
        unassigned = sorted(
            set(self._installed_kernel_types) - {k for ks in interface_keys.values() for k in ks}
        )
        if unassigned:
            raise ValueError(f"{name} keys implement none of its kernel interfaces: {unassigned}")
        return MappingProxyType(interface_keys)

    def key_for(self, interface: str, call: CallSpec) -> str:
        """Return the key of the one implementation of *interface* that serves *call*.

        A key is available where its implementation can run, and it applies where the
        implementation's ``refusal`` accepts the call. Among the available keys that apply,
        the answer is the unique one no other is preferred over: ``preferred_over`` orders
        them, transitively, and the general one is below every other. Neither the order of
        the keys nor any numeric priority decides it.

        Raises:
            OpNotAvailableError: No implementation serves the call.
            RuntimeError: Several implementations serve the call and none is preferred over
                the others.
        """
        device = getattr(call, "device", None) or current_device()
        on_device = False
        rules: dict[str, type[Kernel]] = {}
        rejected: list[str] = []
        for key in self._keys_by_interface[interface]:
            rule = self._installed_kernel_types[key]
            on_device = on_device or (device is not None and device.type in rule.devices)
            reason = rule.unavailable(call) or rule.refusal(call)
            if reason is None:
                rules[key] = rule
            else:
                rejected.append(f"{key} ({rule.__name__}: {reason})")
        chosen = [
            key
            for key, rule in rules.items()
            if not any(
                key in self._preferred_over.get(other, ())
                or (rule.general and not rules[other].general)
                for other in rules
                if other != key
            )
        ]
        if len(chosen) == 1:
            return chosen[0]
        if not on_device and rejected:
            where = "a call that names no device" if device is None else str(device)
            raise OpNotAvailableError(
                f"{type(self).__name__}'s in-tree kernels do not run on {where}; known "
                f"targets for this op: {registered_targets(type(self).__name__)}"
            )
        if not chosen:
            raise OpNotAvailableError(
                "no implementation serves this call: "
                + "; ".join(rejected or ["no implementation is installed"])
                + f". Call: {call}"
            )
        raise RuntimeError(
            f"dispatch is ambiguous: {', '.join(chosen)} all serve this call, and none is "
            f"preferred over the others. Overlapping implementations state which one wins "
            f"through ``preferred_over``, and an interface has at most one general one. "
            f"Call: {call}"
        )

    def kernel_for(self, interface: str, call: Callable[[], CallSpec]) -> object:
        """Return the in-tree entry that serves this call for *interface*, built on a miss.

        The one way an op's in-tree implementation reaches a kernel. The plan of the call in
        progress holds each request's call spec, so *call* runs on the specialization's first
        call only; outside a call, such as in a traced ``forward``, it runs every time.

        Args:
            interface: The kernel interface, one of ``interfaces``.
            call: A function of no argument returning the call spec, an instance of the
                interface's ``request``. What it reads is part of the specialization: the
                call's tensors' layouts, the op's params, the execution arguments and the
                call device. An op with a call-time scalar argument runs it on every request,
                since the scalar's value may decide the spec.

        Raises:
            RuntimeError: What :meth:`_get_or_build_entry` raises.
            TypeError: What :meth:`_get_or_build_entry` raises.
            OpNotAvailableError: This op declares no such interface, so it has nothing in
                tree to serve the call; or no implementation runs on the call's device.
        """
        if interface not in self.interfaces:
            raise OpNotAvailableError(
                f"{type(self).__name__} has no in-tree implementation for {interface!r}, "
                f"so it needs a target that registers one; known targets for this "
                f"op: {registered_targets(type(self).__name__)}"
            )
        if not callable(call):
            raise TypeError(
                f"{type(self).__name__}.kernel_for({interface!r}, ...) takes a function "
                f"returning the call spec, not {type(call).__name__}; the call's plan holds "
                f"what it returns"
            )
        calls = _OPEN_CALLS.stack
        frame = calls[-1] if calls else None
        if frame is None or frame.op is not self or frame.plan is None or frame.plan.scalar:
            return self._entry(interface, call())
        request = self._planned_request(frame, interface, call)
        entries = self._entries_by_call
        resolved = request.resolved
        if resolved is not None and resolved[0] is entries:
            return resolved[1]
        spec = request.spec
        entry = self._entry(interface, spec)
        # A spec naming its device resolves to one entry for as long as the op's entry
        # table stands; one without reads the current device on every request.
        if isinstance(spec, CallSpec) and spec.device is not None and not spec.stated_device_facts:
            request.resolved = (entries, entry)
        return entry

    def call_spec(self, interface: str, call: Callable[[], CallSpec]) -> CallSpec:
        """The call spec *call* builds for a request of *interface*, held by the call's plan as
        :meth:`kernel_for` holds it.

        For a ``forward`` that reads the spec's facts itself, such as a buffer size, before
        passing the spec to ``kernel_for``.
        """
        calls = _OPEN_CALLS.stack
        frame = calls[-1] if calls else None
        if frame is None or frame.op is not self or frame.plan is None or frame.plan.scalar:
            return call()
        return self._planned_request(frame, interface, call).spec

    def _entry(self, interface: str, call: CallSpec) -> object:
        """The entry the op's table holds for *call* under *interface*, built on a miss."""
        if isinstance(call, CallSpec) and call.device is None:
            # A call without a device runs on the accelerator's current one, which the cache
            # key must name.
            device = current_device()
            if device is not None:
                call = call._on_device(device)
        dispatched = self._entries_by_call
        try:
            entry = dispatched.get((interface, call))
        except TypeError:
            # The miss path's checks name what makes the call spec unusable.
            self._get_or_build_entry(interface, call)
            raise
        # A stated device fact takes no part in equality, so a hit would accept it.
        if entry is None or call.stated_device_facts:
            entry = self._get_or_build_entry(interface, call)
            # Bound dispatch memoization independently of compiled-entry reuse.
            if len(dispatched) >= self._MAX_DISPATCHED_CALLS:
                dispatched.pop(next(iter(dispatched)), None)
            dispatched[(interface, call)] = entry
        return entry

    @staticmethod
    def _planned_request(frame: _Frame, interface: str, build: Callable[[], CallSpec]) -> _Request:
        """The plan's record of this kernel request.

        A request matches the plan's request at the same position by interface and by the code
        of *build*; a mismatch drops the plan's requests from there on. *build* runs only for a
        request the plan lacks.
        """
        plan, index = frame.plan, frame.request_index
        frame.request_index = index + 1
        code = getattr(build, "__code__", build)
        requests = plan.requests
        if index < len(requests):
            request = requests[index]
            if request.interface == interface and request.code is code:
                return request
        with plan.filling:
            if index < len(requests):
                request = requests[index]
                if request.interface == interface and request.code is code:
                    return request
                del requests[index:]
            request = _Request(interface, code, build())
            requests.append(request)
        return request

    def _get_or_build_entry(self, interface: str, call: CallSpec) -> object:
        """Get the entry serving *call* for *interface*, building it, on a miss of the call cache.

        The device facts are read from the call's device here, by selection and the
        builder, and the builder runs with that device current and compiles for its build
        target. Two call specs whose implementation names one build identity on one build
        target share one entry. Tuning acts on the entry, never through the builder.

        Raises:
            TypeError: *call* is not the interface's ``request`` type, holds a field that
                cannot key the cache, or states a device fact.
            RuntimeError: What :meth:`key_for` raises.
            OpNotAvailableError: What :meth:`key_for` raises.
        """
        request = self.interfaces[interface].request
        if not isinstance(call, request):
            raise TypeError(
                f"{type(self).__name__}.{interface} takes a {request.__name__} call spec, "
                f"not {type(call).__name__}"
            )
        call._refuse_unkeyable()
        if call.stated_device_facts:
            raise TypeError(
                f"{type(self).__name__}.{interface} reads the device facts from the call's "
                f"device; this call spec states {sorted(call.stated_device_facts)}"
            )
        cls = self._installed_kernel_types[self.key_for(interface, call)]
        identity, build = cls.entry_for(call)
        entries = self._built_entries.setdefault(interface, {})
        # Implementation identity alone need not distinguish incompatible targets.
        key = (cls, identity, target_of(call.device))
        entry = entries.get(key)
        with building_on(call.device):
            if entry is None:
                # Calls missing together build one entry, the one tuning reaches.
                with self._building:
                    entry = entries.get(key)
                    if entry is None:
                        entry = build()
                        entries[key] = entry
            if self._tune_requested:
                # A tuner allocates its candidates' inputs and rebuilds the program, so it
                # runs on the call's device like the build it acts on.
                for kernel in self._entry_kernels(entry):
                    kernel.request_tune()
        return entry

    @classmethod
    @functools.cache
    def _declared_overlaps(cls) -> "Mapping[str, tuple[str, ...]]":
        """The inputs each written slot may share bytes with, by slot: an input's own
        ``overlaps``, and the buffered output's under ``out``."""
        sig = cls._signature.sig
        declared = {n: t.overlaps for n, t in sig.inputs.items() if t.overlaps}
        for t in sig.outputs.values():
            if t.buffer and t.overlaps:
                declared["out"] = t.overlaps
        return MappingProxyType(declared)

    @classmethod
    @functools.cache
    def _forward_io(cls) -> "tuple[tuple[str, ...], frozenset[str]]":
        """The ``forward`` inputs a target is called with, and which of them it writes.

        The names are ``signature.inputs``; the written ones are every input some branch
        writes. A call's own set is ``SignatureCall.written``.
        """
        sig = cls._signature.sig
        written = frozenset(n for n, t in sig.inputs.items() if t.mutated or t.write_only)
        return tuple(sig.inputs), written

    @classmethod
    @functools.cache
    def _forward_parameters(cls) -> inspect.Signature:
        """``forward``'s signature, read once per class: a target call binds it every time."""
        return inspect.signature(cls.forward)

    def _split_forward_args(
        self, args: tuple, kwargs: dict
    ) -> "tuple[tuple, dict[str, torch.Tensor], dict[str, object]]":
        """Split a ``forward`` call into its manifest inputs, written buffers and execution
        arguments.

        The inputs come back in manifest order, an absent optional one as ``None``. A
        tensor ``forward`` takes beyond them is a caller-supplied output buffer, such as
        ``out``; every other argument is an execution argument, by name.
        """
        bound = self._forward_parameters().bind(self, *args, **kwargs)
        bound.apply_defaults()
        names, _ = self._forward_io()
        arguments = bound.arguments
        inputs = tuple(arguments.get(name) for name in names)
        writes, execution = {}, {}
        for name, value in arguments.items():
            if name == "self" or name in names:
                continue
            # The check reports a non-tensor `out` by name.
            if isinstance(value, torch.Tensor) or (name == "out" and value is not None):
                writes[name] = value
            elif name != "out":
                execution[name] = value
        return inputs, writes, execution

    def _observe(
        self,
        inputs: tuple,
        writes: "dict[str, torch.Tensor]",
        execution: "dict[str, object]",
        direct: bool = False,
    ) -> "tuple[tuple | None, list | None]":
        """The pair ``(key, addresses)``.

        *key* is this call's specialization key, or None for a call that keeps no plan; with
        *direct*, None also for a call the dispatcher would do more for than run its body.
        *addresses* holds the address of each call tensor, inputs then writes, or is None when
        the key is None or the op writes no tensor slot."""
        return type(self)._signature.observe(self, inputs, writes, execution, direct)

    def _resolve_plan(self, key: "tuple | None", inputs: tuple, writes: dict) -> _Plan:
        """Run the generated checks on this call and hold what they decide under *key*.

        Construction's checks run again first, for the params as they are now.

        Raises:
            CheckError: The call is outside the signature; nothing is held.
        """
        # Calls resolving one specialization together resolve it once.
        with self._building:
            plan = self._plans.get(key) if key is not None else None
            return plan if plan is not None else self._resolved_plan(key, inputs, writes)

    def _resolved_plan(self, key: "tuple | None", inputs: tuple, writes: dict) -> _Plan:
        """:meth:`_resolve_plan` for a key no plan is held under yet."""
        signature = type(self)._signature
        self._construction_indices = type(self)._check_construction(self)
        tensors = {**dict(zip(signature.sig.inputs, inputs, strict=True)), **writes}
        call = signature.check(self, tensors)
        # Two elements of a written tensor at one address would be written twice. The layout
        # decides it, so the plan, keyed by layout, checks it once.
        for name in sorted(call.written.union(writes)):
            written = tensors[name]
            if isinstance(written, torch.Tensor) and written.numel() and _overlaps_itself(written):
                raise CheckError(f"{type(self).__name__}: {name} must not overlap itself in memory")
        present = signature.states(tensors)
        slots = (*inputs, *writes.values())
        passed = leaves(slots)
        plan = _Plan(
            signature,
            call,
            # A tensor whose storage has no address passes as it is.
            tuple(
                i
                for i, t in enumerate(passed)
                if (address := _address(t)) is not None and address % VECTOR_ACCESS_BYTES
            ),
            _overlap_checks(
                type(self).__name__,
                _leaf_names(tuple(tensors), slots),
                passed,
                call.written.union(writes),
                self._declared_overlaps(),
            ),
            signature.effect_branch(self, present),
            self._writes_no_elements(call),
            self._empty_stages(),
        )
        if key is not None:
            plans = self._plans
            if len(plans) >= self._MAX_PLANS:
                plans.pop(next(iter(plans), None), None)
            plans[key] = plan
        return plan

    def _execute(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        writes: "dict[str, torch.Tensor]",
        written: "frozenset[str] | None" = None,
        execution: "dict[str, object] | None" = None,
        key: object = _UNOBSERVED,
        plan: "_Plan | None" = None,
        addresses: "tuple | None" = None,
    ) -> object:
        """Run one eager call on whichever set of kernels serves this instance.

        Every eager call runs here. ``forward`` gets *inputs* in manifest order, a
        ``kw_only`` one by name, then *writes* and *execution* by name; a target runs the
        whole op. Every call checks its own result and keeps its own record; the plan holds
        only decisions. *written* names the inputs this operator's kernel writes, when that is
        not every input the call writes: an op that registers an inplace companion writes
        nothing through its default operator. A caller that already observed the call passes
        its *key*, *plan* and the *addresses* it observed.

        Raises:
            CheckError: A tensor the call writes shares a byte with another call tensor;
                nothing has run.
            OpNotAvailableError: What :meth:`_bind_target` raises.
        """
        if key is _UNOBSERVED:
            key, addresses = self._observe(inputs, writes, execution or {})
            plan = None if key is None else self._plans.get(key)
        frame = _Frame(self, plan)
        calls = _OPEN_CALLS.stack
        calls.append(frame)
        # Whether this call selects the target. Only that call undoes the selection when it
        # fails; a call that fails before selecting leaves an earlier binding, and the sub-ops'
        # bindings, as they are.
        bound_here = False
        try:
            if plan is None:
                plan = frame.plan = self._resolve_plan(key, inputs, writes)
            if plan.overlaps:
                if addresses is None:
                    # A call no observer read.
                    addresses = list(map(_address, leaves((*inputs, *writes.values()))))
                for i, j, span, other, layout, other_layout, message in plan.overlaps:
                    a, b = addresses[i], addresses[j]
                    if a is None or b is None or not (a < b + other and b < a + span):
                        continue
                    if not _apart(a, layout, b, other_layout):
                        raise CheckError(message)
            signature = type(self)._signature
            values = signature.values.get(plan.call.branch_key)
            if values is not None:
                values(
                    self, dict(zip(signature.sig.inputs, inputs, strict=True)), plan.call.indices
                )
            if plan.contents is not None and self.validate_inputs:
                plan.contents(plan.call, inputs)
            if plan.empty:
                # An empty call runs no implementation, so none has to be available for it;
                # its result is built here, so no result check runs. A named target must exist.
                if self._target_builder is _UNRESOLVED and self.target is not None:
                    select_target(self.target, None)
                result = self._empty_result(plan.call, inputs, writes)
            else:
                caller_inputs = inputs
                kernel_written = plan.call.written if written is None else written
                back = ()
                if plan.copied:
                    inputs, writes, back = _aligned(self, plan, inputs, writes, kernel_written)
                builder = self._target_builder
                if builder is _UNRESOLVED:
                    # Set before selecting, so a selection that fails halfway is undone too.
                    bound_here = True
                    builder = self._bind_target(plan.call.device)
                if builder is not None:
                    result = self._run_on_target(
                        builder,
                        inputs,
                        writes,
                        kernel_written,
                        execution,
                        plan.call.device,
                    )
                elif plan.positional and not writes and not execution:
                    result = self.forward(*inputs)
                else:
                    args, keywords = type(self)._signature.sig.forward_arguments(inputs)
                    result = self.forward(*args, **keywords, **writes, **(execution or {}))
                plan.result_check(result, inputs, writes, self)
                if back:
                    result = _restored(result, back)
                    for copy, caller in back:
                        caller.copy_(copy)
                inputs = caller_inputs
            calls.pop()
            self._commit(plan, inputs, frame.children, calls)
            return result
        except Exception:
            if calls and calls[-1] is frame:
                calls.pop()
            # The call that bound the target unbinds it. A failure out of a compiled graph
            # reaches no handler of ``__call__``, so this one is the only one.
            if bound_here:
                self._reset_binding()
            raise

    @classmethod
    @functools.cache
    def _empty_stages(cls) -> "Mapping[str, tuple]":
        """The stage mapping of a call that collected nothing: every declared stage, empty."""
        return MappingProxyType({stage: () for stage in cls.delegate_types})

    def _commit(self, plan: _Plan, inputs: tuple, children: list, calls: list) -> None:
        """Keep a completed call of *plan* as the last one, with the records of the calls its
        sub-ops completed inside it, by stage and in completion order
        (docs/design/roofline.md §2.2), and report it to the innermost of *calls*.

        Raises:
            RuntimeError: A sub-op this op does not hold through :meth:`delegate_for`
                completed a call during it. Its stage is unknown, so ``stages`` would miss
                it and the roofline priced from them would be wrong.
        """
        staged = []
        for op, record in children:
            stage = self._delegate_stages.get(id(op))
            if stage is None:
                raise RuntimeError(
                    f"{type(self).__name__}: {type(op).__name__} completed a call inside "
                    f"this call but is not held through delegate_for, so no stage records "
                    f"it; hold every sub-op with delegate_for(stage, identity, ...)"
                )
            staged.append((stage, record))
        metadata = {name: inputs[i] for name, i in plan.metadata} if plan.metadata else {}
        record = _CallRecord(plan.call, metadata, tuple(staged), plan.stages)
        self._last_call = record
        if calls:
            calls[-1].children.append((self, record))

    def _complete_unplanned(self, call: object) -> None:
        """Keep *call*, which completed with no plan and no sub-op call, as the last one: an
        eager call on meta tensors, which completes in its operator's fake."""
        # Its own copy of the indices, as a plan keeps: the op's later edits must not reach it.
        held = dataclasses.replace(call, indices=copy.deepcopy(call.indices))
        record = _CallRecord(held, call.metadata, (), self._empty_stages())
        self._last_call = record
        calls = _OPEN_CALLS.stack
        if calls:
            calls[-1].children.append((self, record))

    def _run_on_target(
        self,
        builder: object,
        inputs: "tuple[torch.Tensor | None, ...]",
        writes: "dict[str, torch.Tensor]",
        written: "frozenset[str] | None" = None,
        execution: "dict[str, object] | None" = None,
        device: "torch.device | None" = None,
    ) -> object:
        """Run the whole op on the target whose *builder* this call selected.

        Every target is guaranteed: every tensor on one device, every input the call does not
        write contiguous, every call tensor on a 16-byte boundary, and the signature's checks.
        The kernel is built once per builder, device, input specs and manifest params. *device*
        is the checked call device; a traced call, which runs no check, passes None.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        names, mutated = self._forward_io()
        written = mutated if written is None else written
        inputs = tuple(
            type(t)(x.contiguous() for x in t)
            if isinstance(t, (list, tuple)) and name not in written
            else t.contiguous()
            if isinstance(t, torch.Tensor) and name not in written
            else t
            for name, t in zip(names, inputs, strict=True)
        )
        if device is None:
            device = self._traced_device(leaves((*inputs, *writes.values())))
        params = self._manifest_params()
        # The builder's description of each input keys the kernel: an absent input keeps its
        # slot as ``None``, so two calls differing only in which one they omit key apart, and a
        # call-time scalar keys by its type, never its value. The builder keys too: a kernel
        # serves only the builder that built it.
        specs = tuple(map(_slot_spec, inputs))
        signature = (device, *specs, _snapshot(params), builder)
        built = self._target_kernels
        kernel = built.get(signature)
        if kernel is None:
            # Concurrent misses build one kernel, from copies of the params.
            with self._building:
                kernel = built.get(signature)
                if kernel is None:
                    owned = copy.deepcopy(params)
                    kernel = built[signature] = self._build_target_kernel(
                        builder, device, specs, owned
                    )
        result = kernel(*inputs, **writes, **(execution or {}))
        # An op whose every output is an input it writes returns nothing, as its forward does.
        outputs = type(self)._signature.sig.outputs
        if outputs and all(name in names for name in outputs):
            return None
        return result

    def _traced_device(self, arguments: tuple) -> "torch.device | None":
        """The call device of a call no check placed, from its *arguments*: its tensors'
        device, `device: cpu` tensors aside, else the declared ``device`` param, else the
        accelerator's current one, as the generated checks place it."""
        devices = {t.device for t in arguments if isinstance(t, torch.Tensor)}
        devices = {d for d in devices if d.type != "cpu"} or devices
        return next(iter(devices)) if devices else tensorless_device(self)

    def _build_target_kernel(
        self,
        builder: object,
        device: "torch.device | None",
        specs: tuple,
        params: "dict[str, object]",
    ) -> object:
        """Ask *builder*, with *device* current, for the kernel serving calls whose inputs
        *specs* describe, one :func:`_slot_spec` each, built with the manifest *params*.

        In tuned mode the kernel is tuned before it is returned, so one whose tuning raised is
        never cached.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        with device_context(device):
            kernel = builder(*specs, **params)
        if not callable(kernel):
            raise OpNotAvailableError(
                f"the kernel builder {builder!r} returned {kernel!r}, which is not "
                f"callable, for {type(self).__name__}; a builder returns something the "
                f"op can call with the tensors it was described"
            )
        if self._tune_requested:
            self._tune_target_kernel(device, kernel)
        return kernel

    def _tune_target_kernel(self, device: "torch.device | None", kernel: object) -> None:
        """Put a kernel this op's target built for *device* in tuned mode, on that device.

        A kernel that defines no ``request_tune`` cannot be tuned, and the op warns once.
        """
        tune = getattr(kernel, "request_tune", None)
        if callable(tune):
            with device_context(device):
                tune()
        elif not self._tune_warned:
            self._tune_warned = True
            warnings.warn(
                f"{type(self).__name__}'s target kernel {kernel!r} defines no request_tune",
                UserWarning,
                stacklevel=3,
            )

    def _manifest_params(self) -> dict[str, object]:
        """The op's manifest params, by name, with the values this instance holds: what
        ``build_kernel`` is called with by keyword.

        Raises:
            AttributeError: The op keeps a manifest param under another name.
        """
        names = getattr(self, PARAM_NAMES_ATTRIBUTE, None)
        if names is None:
            return {}
        values = {}
        for param in names:
            try:
                values[param] = getattr(self, param)
            except AttributeError:
                raise AttributeError(
                    f"{type(self).__name__} declares manifest param {param!r} but keeps no "
                    f"attribute of that name; a backend is called with the manifest's "
                    f"names, so this op has to store it under one"
                ) from None
        return values

    def _entries_built(self, interface: str) -> Mapping[Hashable, object]:
        """A read-only mapping of the entries built for *interface* so far, for introspection
        only: a live view for the in-tree kernels, and for an op a target serves a copy of the
        target's kernels, one per input signature, whatever the interface.
        """
        builder = self._target_builder
        if builder is not None and builder is not _UNRESOLVED:
            # Copied in one step, so a concurrent call adding a kernel cannot break the loop.
            held = tuple(self._target_kernels.items())
            return MappingProxyType(dict(held))
        return MappingProxyType(self._built_entries.get(interface, {}))

    def delegate_for(
        self, stage: str, identity: Hashable, given: "Op | None" = None, /, **params: object
    ) -> "Op":
        """Return the sub-op held for *stage* under *identity*, building it on a miss.

        The one way an op holds a sub-op. A miss builds ``delegate_types[stage](**params)``
        with this op's ``target``; *given*, an implementation the caller injected for the
        stage, is held instead. A sub-op first held while this op is in tuned mode, built or
        given, is put in tuned mode too. Eager only, like :meth:`kernel_for`: a traced
        ``forward`` reaches no miss.

        Args:
            stage: A key of ``delegate_types``.
            identity: Everything that can change what gets built, as an implementation's
                ``entry_for`` returns it, or ``None`` to hold the sub-op for this op's
                specialization (its params). Under ``None`` a sub-op does not see a param
                edited in place (assign a new value).
            given: The caller's implementation for this stage, or ``None`` to build one.
                One given under identity ``None`` is held for every specialization.
            params: The sub-op's constructor arguments other than the execution policy.

        Raises:
            KeyError: *stage* is not declared in ``delegate_types``.
            ValueError: *given* is already held under another stage or identity.
        """
        cls = self.delegate_types[stage]
        entries = self._delegates.setdefault(stage, {})
        if identity is None and given is None and None not in entries:
            # Held by this op's specialization, which the plan of the call in progress fixes.
            calls = _OPEN_CALLS.stack
            frame = calls[-1] if calls else None
            plan = frame.plan if frame is not None and frame.op is self else None
            if plan is None:
                return self._settled_delegate(stage, params)
            # By request position: one plan's forward makes the same requests in order.
            held, index = plan.delegates, frame.delegate_index
            frame.delegate_index = index + 1
            if index < len(held):
                entry = held[index]
                if entry[0] == stage:
                    return entry[1]
            with plan.filling:
                delegate = self._settled_delegate(stage, params)
                del held[index:]
                held.append((stage, delegate))
            return delegate
        if identity not in entries:
            # Concurrent misses, from any plan, build one sub-op.
            with self._building:
                if identity not in entries:
                    delegate = given if given is not None else cls(**params, target=self.target)
                    # A sub-op's completed calls are filed under the one stage that holds it.
                    if id(delegate) in self._delegate_stages:
                        raise ValueError(
                            f"{type(self).__name__} already holds this {type(delegate).__name__} for "
                            f"stage {self._delegate_stages[id(delegate)]!r}; one sub-op is held under "
                            f"one (stage, identity), so it cannot also be held for {stage!r} under "
                            f"{identity!r}"
                        )
                    entries[identity] = delegate
                    # Which stage holds each sub-op, replaced ones included, until it is
                    # collected.
                    self._delegate_stages[id(delegate)] = stage
                    weakref.finalize(
                        delegate, self._delegate_stages.pop, id(delegate), None
                    ).atexit = False
                    # A sub-op held while this op is in tuned mode joins it, built or given.
                    if self._tune_requested:
                        delegate.request_tune()
        return entries[identity]

    def _settled_delegate(self, stage: str, params: dict) -> "Op":
        """The sub-op of *stage* for this op's params, built from *params* on a miss."""
        signature = type(self)._signature
        identity = (stage, None if signature is None else param_state(self, signature.sig))
        return self.delegate_for(stage, identity, **params)

    def _held_delegates(self) -> Sequence["Op"]:
        """Return the sub-ops this op holds, in stage order, then in the order they were built.

        Derived from what :meth:`delegate_for` holds; an op does not override it.
        """
        held = self._delegates
        return tuple(op for stage in self.delegate_types for op in held.get(stage, {}).values())

    @property
    def serving_target(self) -> Target:
        """The target that serves this instance: ``None`` until a call selects it, and again
        if that call fails; ``BUILTIN`` for the in-tree implementation however it was chosen;
        otherwise the target's name.
        """
        if self._target_builder is _UNRESOLVED:
            return None
        if self._target_builder is None:
            return BUILTIN
        return self._serving_target

    def kernel_config(self) -> dict[str, Any] | None:
        """The op's own ``config``, else the first ``config`` an in-tree or target kernel
        published, or ``None``."""
        own = getattr(self, "config", None)
        if own:
            return own
        kernels = (
            *self.iter_kernels(),
            *(kernel for _, _, kernel in self._target_kernels_by_device()),
        )
        for kernel in kernels:
            config = getattr(kernel, "config", None)
            if config:
                return config
        return None

    def iter_kernels(self) -> Iterator[Kernel]:
        """Yield every in-tree ``Kernel`` the entries of this op and its held sub-ops hold,
        each one once. A target's kernels are not yielded.
        """
        seen: set[int] = set()
        for op in self._walk_ops():
            # Copied in one step each: calls may add entries while this walks.
            held = [
                entry
                for entries in tuple(op._built_entries.values())
                for entry in tuple(entries.values())
            ]
            for entry in held:
                for kernel in self._entry_kernels(entry):
                    if id(kernel) not in seen:
                        seen.add(id(kernel))
                        yield kernel

    def _target_kernels_by_device(self) -> Iterator[tuple["Op", "torch.device | None", object]]:
        """Yield each kernel a target built for this op or the ops it runs, with the op that
        holds it and the device it was built for.

        One kernel a target returned for several input signatures on one device is yielded
        once, and once per device it was built for: tuning acts per device.
        """
        seen: set[tuple] = set()
        for op in self._walk_ops():
            # Copied in one step: tuning runs between yields while calls may add kernels.
            for signature, kernel in tuple(op._target_kernels.items()):
                device = signature[0]
                if (device, id(kernel)) not in seen:
                    seen.add((device, id(kernel)))
                    yield op, device, kernel

    @staticmethod
    def _entry_kernels(entry: object) -> "list[Kernel]":
        """Return the kernels one entry holds.

        An entry is a kernel, a sequence of kernels built together, or a dataclass
        carrying them alongside what else the specialization implies. An entry that
        hides its kernels from this walk is invisible to ``request_tune``.
        """
        if isinstance(entry, Kernel):
            return [entry]
        if isinstance(entry, (tuple, list)):
            return [k for item in entry for k in Op._entry_kernels(item)]
        if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
            return [
                k
                for f in dataclasses.fields(entry)
                for k in Op._entry_kernels(getattr(entry, f.name))
            ]
        return []

    def _walk_ops(self) -> Iterator["Op"]:
        """Yield this op and the ops it runs kernels through, each one once."""
        seen: set[int] = set()
        stack: list[Op] = [self]
        while stack:
            op = stack.pop()
            if id(op) in seen:
                continue
            seen.add(id(op))
            yield op
            stack.extend(reversed(op._held_delegates()))

    def request_tune(self) -> None:
        """Put the op and its sub-ops in tuned mode: the kernels built so far, and every kernel
        built later, specializations not seen yet included.

        A target's kernel is tuned once per device it was built for, with that device current;
        the op warns once for a target kernel that defines no ``request_tune()``. Tuning a
        kernel is idempotent. A kernel whose tuning raises leaves the op in tuned mode, so
        calling this again tunes what was not reached.
        """
        for op in self._walk_ops():
            op._tune_requested = True
        for kernel in self.iter_kernels():
            kernel.request_tune()
        for op, device, kernel in self._target_kernels_by_device():
            op._tune_target_kernel(device, kernel)

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run the op.

        Returns:
            The op's output tensor, or a tuple of them in the order its manifest entry
            declares.
        """
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run one call through :meth:`_execute`, via the compile boundary when the op has one.

        Traced, an op without a compile boundary runs no check: it calls its target's kernel,
        or ``forward``.

        Raises:
            TypeError: The op has no manifest entry.
        """
        if self.compile_op_names:
            return self._call_boundary(*args, **kwargs)
        if type(self)._signature is None:
            raise self._no_entry()
        if torch.compiler.is_compiling():
            builder = self._target_builder
            if builder is _UNRESOLVED:
                builder = self._bind_target(self._traced_device(leaves((*args, *kwargs.values()))))
            if builder is not None:
                inputs, writes, execution = self._split_forward_args(args, kwargs)
                return self._run_on_target(builder, inputs, writes, None, execution)
            return self.forward(*args, **kwargs)
        inputs, writes, execution = self._split_forward_args(args, kwargs)
        return self._execute(inputs, writes, None, execution)

    @staticmethod
    def _writes_no_elements(call: object) -> bool:
        """Whether every tensor *call* writes, its outputs and written inputs, holds no elements.

        Such a call has nothing to compute, so no implementation runs it. The written tensors
        decide, not the inputs: an empty input whose output still holds elements, such as a
        sum over an empty axis, runs as any other call does.
        """
        written = False
        for t, _, writes in call.traffic:
            if writes:
                if math.prod(call.tensors[t][0]):
                    return False
                written = True
        return written

    def _empty_result(
        self, call: object, inputs: tuple, writes: "dict[str, torch.Tensor]"
    ) -> object:
        """What ``forward`` returns for a call that writes nothing, computed from the signature.

        An absent output is ``None``, a buffered one is the ``out`` passed, one aliasing a
        written input is that input, and every other one is a new tensor of the checked shape
        and dtype on the call device.
        """
        sig = type(self)._signature.sig
        named = dict(zip(sig.inputs, inputs, strict=True))
        values = []
        for name, decl in sig.outputs.items():
            if name not in call.tensors:
                values.append(None)
            elif decl.alias in call.written:
                values.append(named[decl.alias])
            elif decl.buffer and call.has_out:
                values.append(writes["out"])
            else:
                shape, dtype = call.tensors[name]
                values.append(torch.empty(shape, dtype=getattr(torch, dtype), device=call.device))
        return None if not values else values[0] if len(values) == 1 else tuple(values)

    def _reset_binding(self) -> None:
        """Reset the binding group to what ``__init__`` built, when the call that bound the
        target did not finish; every kernel built under that target is dropped.

        The sub-ops are reset with it, so none keeps what the failed call bound.
        """
        for delegate in self._held_delegates():
            delegate._reset_binding()
        self._target_builder = _UNRESOLVED
        self._serving_target = None
        self._entries_by_call = {}
        self._built_entries = {}
        self._target_kernels = {}

    def _bind_target(self, device: "torch.device | None") -> object:
        """Decide which target serves this instance on the call *device*, remember its
        builder, and return it: ``None`` for the in-tree kernels.

        The decision is final. Without a named or process-default target, a call with no
        device decides nothing. A composite without a builder lets each sub-op select its own.

        Raises:
            OpNotServedError: The selected target registers no builder for this op,
                and the op builds kernels of its own.
        """
        target = select_target(self.target, device)
        if target is None:
            self._serving_target = None
            if device is not None:
                self._target_builder = None  # a device was probed, so the answer is decided
            return None
        if target is BUILTIN:
            self._serving_target = BUILTIN
            self._target_builder = None
            return None
        # A target serves only the ops of tileops, which it knows by their manifest keys.
        builder = (
            registered_kernel_builder(type(self).__name__, target) if in_tree(type(self)) else None
        )
        if builder is None and not self.kernel_types:
            self._serving_target = target
            self._target_builder = None
            return None
        if builder is None:
            raise OpNotServedError(
                f"target {target!r} registers no kernel builder for "
                f"{type(self).__name__}; targets that do: "
                f"{registered_targets(type(self).__name__)}. There is no fall back to the "
                f"in-tree implementation: those kernels do not run on this target's "
                f"devices.",
                target=target,
            )
        self._serving_target = target
        self._target_builder = builder
        return builder
