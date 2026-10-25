import copy
import dataclasses
import functools
import inspect
import math
import threading
import warnings
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Callable,
    ClassVar,
    Hashable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._ops._params_codegen import PARAM_NAMES_ATTRIBUTE
from tileops._ops._signature_codegen import (
    ResultCheck,
    SignatureCall,
    _snapshot,
    in_tree,
    param_state,
)
from tileops._ops.compile_boundary import register_instance
from tileops._utils.device import current_device, device_context
from tileops.backend import (
    BUILTIN,
    OpNotAvailableError,
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


# The calls in progress on this thread, innermost last: each one's frame.
_OPEN_CALLS = threading.local()

# ``Op._execute``'s *key* when its caller has not observed the call.
_UNOBSERVED = object()


class _Plan:
    """What one specialization of an op's calls decides, resolved on its first call.

    A specialization is everything the generated checks and the call specs read: each tensor's
    shape, strides, dtype and device, which tensors are present, the op's params and
    construction-time tensors, the execution arguments, and the current device of a call no
    tensor places. The plan holds the checked call's facts, its effect branch, its result
    check, whether it writes no element, and each kernel request ``forward`` made, in request
    order. It holds no tensor a call passes, returns or completes.
    """

    __slots__ = (
        "call",
        "metadata",
        "effect",
        "empty",
        "result_check",
        "stages",
        "positional",
        "requests",
        "delegates",
        "filling",
    )

    def __init__(self, signature, call, effect: tuple, empty: bool, stages) -> None:
        """Hold what *signature*'s checks decided for *call*, without its metadata tensors;
        *stages* is the op's stage mapping of a call that collected nothing."""
        sig = signature.sig
        order = list(sig.inputs)
        # Its own copy of the indices: a param a call reads can be a list or an ADT, which
        # the op's later edits must not reach.
        self.call = dataclasses.replace(call, indices=copy.deepcopy(call.indices), metadata=None)
        # Each metadata tensor by name, with its position among the inputs, or None for a
        # construction-time tensor, which the op holds.
        self.metadata = tuple(
            (name, order.index(name) if name in sig.inputs else None) for name in call.metadata
        )
        self.effect, self.empty, self.stages = effect, empty, stages
        self.result_check = ResultCheck(sig, call)
        self.positional = signature.positional
        # Each kernel request ``forward`` made, in request order.
        self.requests: list = []
        # Each ``(stage, sub-op)`` ``forward`` held under no identity, in request order.
        self.delegates: list = []
        # Held while a call extends either list, so calls missing together extend it once.
        # Reentrant: a sub-op built under it may resolve its own plan.
        self.filling = threading.RLock()


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


def _open_calls() -> list:
    calls = getattr(_OPEN_CALLS, "stack", None)
    if calls is None:
        calls = _OPEN_CALLS.stack = []
    return calls


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
    computation in ``forward``.

    Code that calls an op constructs it with ``target``, the construction parameter every op
    takes, which names the set of kernels that serves it. It calls the op as ``op(...)`` with
    ``forward``'s parameters and never calls ``forward`` itself. ``serving_target`` is the
    target the instance selected; ``last_call`` and ``eval_roofline()`` are the last
    completed call and its ``(flops, bytes)``; ``request_tune()`` and ``kernel_config()``
    request tuning and read the configuration in use; ``iter_kernels()`` lists the kernels
    the op and its sub-ops have built.

    A subclass calls ``Op.__init__(*, target=None)`` once its manifest parameters are
    assigned, and builds derived attributes and sub-ops after it. It declares its
    implementations and the places it calls them in ``kernel_types`` and ``interfaces``, and in
    ``injected_parameters`` the constructor parameters beyond the manifest's and ``target``
    that the manifest validator allows. ``dtype`` is ``None`` here; a subclass taking a
    ``dtype`` parameter overrides it. ``forward`` is the computation, the one method every op
    writes. Inside it, ``kernel_for(interface, call)`` returns the entry that serves a call and
    ``key_for(interface, call)`` the key of its implementation; ``delegate_types`` and
    ``delegate_for(stage, identity, ...)`` declare and hold the sub-ops. ``roof_key()``,
    ``eval_roofline_read_bytes()`` and ``roofline_data_terms()`` are the parts of the roofline
    a subclass may override.

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
    # The target bound with it, for ``serving_target`` and error messages.
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
    # The generated signature plan; None for a class without a manifest entry.
    _signature: ClassVar[object] = None
    dtype: Optional[torch.dtype] = None

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
        from tileops._ops._params_codegen import maybe_install_param_names
        from tileops._ops._signature_codegen import maybe_install_signature, op_entry

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
        maybe_install_param_names(cls, entry)

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
    # Naming them is what lets a test assert the traced graph holds nothing else, which is
    # what keeps the graph the same when another target serves the op. A tuple because a
    # conditional in-place write registers two. Registration happens once per class, so this
    # is class state.
    compile_op_names: ClassVar[tuple[str, ...]] = ()

    # Injected implementation objects ``__init__`` takes beyond ``signature.params`` and the
    # execution-policy parameters every op takes (docs/design/manifest.md § Signature).
    injected_parameters: ClassVar[tuple[str, ...]] = ()

    # A check of the input values the signature cannot state, run on every call that writes
    # an element before a target's kernel or ``forward`` runs: None, or a method taking the
    # inputs in manifest order.
    _check_input_values: ClassVar[Optional[Callable[..., None]]] = None

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
        self._instance_key = register_instance(self)
        # The binding caches and the sub-op tables, so no method asks whether they exist;
        # installing built the two entry caches.
        self._target_kernels = {}
        # Held while an entry is built; reentrant, since a build may resolve another entry.
        self._building = threading.RLock()
        self._delegates = {}
        self._delegate_stages = {}
        self._effect_branches = {}
        self._plans = {}

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

    def eval_roofline_read_bytes(self) -> Optional[int]:
        """The read half of ``eval_roofline()[1]``, for the NCU bytes audit.

        ``bytes`` minus the write half, which the signature settles: every
        declared output once, plus a ``mutated`` input that is not an output.
        An op that reads only part of an input needs no override -- its
        ``bytes`` already counted that part.

        Returns:
            The read half in bytes, or ``None`` from an override whose read half the
            signature cannot settle.
        """
        call = self.last_call
        write_bytes = sum(call.bytes(t) * w for t, _, w in call.traffic)
        return int(self.eval_roofline()[1]) - write_bytes

    def roofline_data_terms(self) -> "dict[str, int]":
        """What decided this call's ``bytes``, where its inputs' values decided it.

        Two calls of one shape can move different amounts -- a routed MoE reads
        the experts its routing selected -- and the benchmark records this
        beside the reading so such a number says why it moved.

        Nothing judges it, and it is not part of ``(flops, bytes)``. Empty
        unless the op's traffic follows its inputs' values.
        """
        return {}

    def roof_key(self) -> str:
        """GPU-profile key of the compute unit that prices this op's FLOPs.

        ``eval_roofline()`` counts the work; ``roof_key()`` names the
        peak that bounds it (docs/design/roofline.md §1.2). The key is a
        statement about the *optimal* implementation, declared by the op
        author — never inferred from the running kernel, so a kernel on the
        wrong unit is still measured against the right ceiling.

        The base default covers ops whose arithmetic runs on CUDA cores in
        fp32 (elementwise, reductions, norms, scans). An op whose FLOPs are
        matmul contractions overrides this with ``tensor_core_roof`` of the
        contraction's input dtype read from ``self.last_call`` (or a
        backend-specific key). Valid whenever ``eval_roofline()`` is.
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
        """Return the in-tree entry that serves this call for *interface*, getting it on a miss.

        The one way an op's in-tree implementation reaches a kernel. It runs only when the
        in-tree kernels serve the op: a target serves the whole op instead
        (:meth:`_run_on_target`). The plan of the call in progress holds the call spec of each
        request, so *call* runs on the specialization's first call only; a request outside a
        call, such as a traced ``forward``'s, runs it every time. A miss selects the
        implementation and builds or reuses the entry its build identity names
        (:meth:`_get_or_build_entry`).

        Args:
            interface: The kernel interface, one of ``interfaces``.
            call: A function of no argument returning the call spec, an instance of the
                interface's ``request``. What it reads is part of the specialization: the
                call's tensors' layouts, the op's params, the execution arguments and the
                call device.

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
        calls = _OPEN_CALLS.__dict__.get("stack")
        frame = calls[-1] if calls else None
        if frame is None or frame.op is not self or frame.plan is None:
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
        calls = _OPEN_CALLS.__dict__.get("stack")
        frame = calls[-1] if calls else None
        if frame is None or frame.op is not self or frame.plan is None:
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
        builder, and the builder runs with that device current. Two call specs whose
        implementation names one build identity share one entry. Tuning acts on the
        entry, never through the builder.

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
        entry = entries.get((cls, identity))
        with device_context(call.device):
            if entry is None:
                # Calls missing together build one entry, the one tuning reaches.
                with self._building:
                    entry = entries.get((cls, identity))
                    if entry is None:
                        entry = build()
                        entries[(cls, identity)] = entry
            if self._tune_requested:
                # A tuner allocates its candidates' inputs and rebuilds the program, so it
                # runs on the call's device like the build it acts on.
                for kernel in self._entry_kernels(entry):
                    kernel.request_tune()
        return entry

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

    @classmethod
    @functools.cache
    def _forward_outputs(cls) -> "tuple[str, ...]":
        """The op's declared outputs, in order."""
        return tuple(cls._signature.sig.outputs)

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

    def _named_tensors(
        self, inputs: "tuple[torch.Tensor | None, ...]", writes: "dict[str, torch.Tensor]"
    ) -> "dict[str, torch.Tensor | None]":
        """Every tensor a call passes, by signature name: what the generated checks read.

        One call builds it once and hands the same mapping to the check and to the result
        check; the two see identical arguments by construction.
        """
        sig = type(self)._signature.sig
        return {**dict(zip(sig.inputs, inputs, strict=True)), **writes}

    def _observe(
        self,
        inputs: tuple,
        writes: "dict[str, torch.Tensor]",
        execution: "dict[str, object]",
        direct: bool = False,
    ) -> "tuple | None":
        """The key of this call's specialization, or None for a call that keeps no plan; with
        *direct*, None also for a call the dispatcher would do more for than run its body."""
        return type(self)._signature.observe(self, inputs, writes, execution, direct)

    def _resolve_plan(
        self, key: "tuple | None", inputs: tuple, writes: dict, execution: dict
    ) -> _Plan:
        """Run the generated checks on this call and hold what they decide under *key*.

        Construction's checks run again first, for the params as they are now. The call checks
        may move a construction-time tensor to the call device and signature dtype, which
        changes the key; the plan is held only when the key observed after the checks is
        *key*.

        Raises:
            CheckError: The call is outside the signature; nothing is held.
        """
        # Calls resolving one specialization together resolve it once.
        with self._building:
            plan = self._plans.get(key) if key is not None else None
            return plan if plan is not None else self._resolved_plan(key, inputs, writes, execution)

    def _resolved_plan(
        self, key: "tuple | None", inputs: tuple, writes: dict, execution: dict
    ) -> _Plan:
        """:meth:`_resolve_plan` for a key no plan is held under yet."""
        signature = type(self)._signature
        self._construction_indices = type(self)._check_construction(self)
        tensors = self._named_tensors(inputs, writes)
        call = signature.check(self, tensors)
        present = tuple(tensors.get(name) is not None for name in signature.tensor_axes)
        plan = _Plan(
            signature,
            call,
            signature.effect_branch(self, present),
            self._writes_no_elements(call),
            self._empty_stages(),
        )
        if key is not None and (
            not signature.ctor_names or self._observe(inputs, writes, execution) == key
        ):
            # Calls whose shapes keep changing, such as decode steps over a growing cache,
            # resolve a plan each; the oldest one goes once this many are held.
            held = 256
            plans = self._plans
            if len(plans) >= held:
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
    ) -> object:
        """Run one eager call on whichever set of kernels serves this instance.

        Every eager call runs here. ``forward`` gets *inputs* in manifest order, a
        ``kw_only`` one by name, then *writes* and *execution* by name; a target runs the
        whole op. Every call checks its own result and keeps its own record; the plan holds
        only decisions. *written* names the inputs this operator's kernel writes, when that is
        not every input the call writes: an op that registers an inplace companion writes
        nothing through its default operator. A caller that already observed the call passes
        its *key* and *plan*.

        Raises:
            OpNotAvailableError: What :meth:`_bind_target` raises.
        """
        if key is _UNOBSERVED:
            key = self._observe(inputs, writes, execution or {})
            plan = None if key is None else self._plans.get(key)
        frame = _Frame(self, plan)
        calls = _OPEN_CALLS.__dict__.get("stack") or _open_calls()
        calls.append(frame)
        # Whether this call selects the target. Only that call undoes the selection when it
        # fails; a call that fails before selecting leaves an earlier binding, and the sub-ops'
        # bindings, as they are.
        bound_here = False
        try:
            if plan is None:
                plan = frame.plan = self._resolve_plan(key, inputs, writes, execution or {})
            if plan.empty:
                # An empty call runs no implementation, so none has to be available for it;
                # its result is built here, so no result check runs. A target it names must
                # still exist.
                if self._target_builder is _UNRESOLVED and self.target is not None:
                    select_target(self.target, None)
                result = self._empty_result(plan.call, inputs, writes)
            else:
                check = self._check_input_values
                if check is not None:
                    check(*inputs)
                builder = self._target_builder
                if builder is _UNRESOLVED:
                    # Set before selecting, so a selection that fails halfway is undone too.
                    bound_here = True
                    self._bind_target(plan.call.device)
                    builder = self._target_builder
                if builder is not None:
                    result = self._run_on_target(
                        inputs,
                        writes,
                        plan.call.written if written is None else written,
                        execution,
                        plan.call.device,
                    )
                elif plan.positional and not writes and not execution:
                    result = self.forward(*inputs)
                else:
                    args, keywords = type(self)._signature.sig.forward_arguments(inputs)
                    result = self.forward(*args, **keywords, **writes, **(execution or {}))
                plan.result_check(result, inputs, writes, self)
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
        metadata = (
            {name: getattr(self, name) if i is None else inputs[i] for name, i in plan.metadata}
            if plan.metadata
            else {}
        )
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
        calls = _open_calls()
        if calls:
            calls[-1].children.append((self, record))

    def _has_target_builder(self) -> bool:
        """Whether a target's builder, rather than the in-tree kernels, serves this instance."""
        return self._target_builder is not None and self._target_builder is not _UNRESOLVED

    def _run_on_target(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        writes: "dict[str, torch.Tensor]",
        written: "frozenset[str] | None" = None,
        execution: "dict[str, object] | None" = None,
        device: "torch.device | None" = None,
    ) -> object:
        """Run the whole op on the target bound to this instance.

        What the op layer guarantees every target: every tensor on one device, every
        input the call does not write contiguous, and the checks generated from the signature.
        The kernel is built once per device, input dtypes and shapes, and the manifest params
        it is built with, a tensor param by identity.

        *device* is the call device the checks placed the call on; a traced call, which runs
        no check, places it from its tensors or the ``device`` param.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        names, mutated = self._forward_io()
        written = mutated if written is None else written
        inputs = tuple(
            t if t is None or name in written else t.contiguous()
            for name, t in zip(names, inputs, strict=True)
        )
        if device is None:
            device = self._traced_device((*inputs, *writes.values()))
        params = self._manifest_params()
        # An absent optional input keeps its slot as ``None``, so two calls differing
        # only in which one they omit key apart. A tensor param keys by identity, which a
        # builder may hold: the key holds the tensor too, so no other one takes its identity.
        tensors = tuple(v for v in params.values() if isinstance(v, torch.Tensor)) if params else ()
        signature = (
            device,
            *(None if t is None else (t.dtype, tuple(t.shape)) for t in inputs),
            _snapshot(params),
            *tensors,
        )
        # A builder may read a tensor param's contents, so a write in place rebuilds.
        versions = (
            tuple(None if t.is_inference() else t._version for t in tensors) if tensors else ()
        )
        named = dict(zip(names, inputs, strict=True))
        # Each kernel with the versions it was built at, read and written as one entry.
        built = self.__dict__.get("_target_built")
        if built is None:
            built = self.__dict__.setdefault("_target_built", {})
        entry = built.get(signature)
        if entry is None or entry[1] != versions or None in versions:
            # Concurrent misses build one kernel, from copies of the non-tensor params.
            with self._building:
                entry = built.get(signature)
                if entry is None or entry[1] != versions or None in versions:
                    owned = {
                        k: v if isinstance(v, torch.Tensor) else copy.deepcopy(v)
                        for k, v in params.items()
                    }
                    kernel = self._build_target_kernel(device, named, owned)
                    entry = (kernel, versions)
                    if None not in versions:
                        built[signature] = entry
                        self._target_kernels[signature] = kernel
        kernel = entry[0]
        result = kernel(*inputs, **writes, **(execution or {}))
        # An op whose every output is an input it writes returns nothing, as its forward does.
        outputs = self._forward_outputs()
        if outputs and all(name in named for name in outputs):
            return None
        return result

    def _traced_device(self, arguments: tuple) -> "torch.device | None":
        """The call device of a call no check placed, from its *arguments*: its tensors'
        device, `device: cpu` tensors aside, else :meth:`_device_without_call_tensors`."""
        devices = {t.device for t in arguments if isinstance(t, torch.Tensor)}
        devices = {d for d in devices if d.type != "cpu"} or devices
        return next(iter(devices)) if devices else self._device_without_call_tensors()

    def _build_target_kernel(
        self,
        device: "torch.device | None",
        inputs: "dict[str, torch.Tensor | None]",
        params: "dict[str, object]",
    ) -> object:
        """Ask the target for the kernel serving calls on *device* described by *inputs*,
        built with the manifest *params*.

        The builder runs with *device* current. In tuned mode the kernel is tuned before it
        is returned, so a kernel whose tuning raised is never cached and the next call builds
        it again.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        specs = tuple(None if t is None else TensorSpec.of(t) for t in inputs.values())
        # Built with its device current, as an in-tree entry is: a builder may compile or
        # allocate for whichever device is current.
        with device_context(device):
            kernel = self._target_builder(*specs, **params)
        if not callable(kernel):
            raise OpNotAvailableError(
                f"target {self._serving_target!r} built {kernel!r} for "
                f"{type(self).__name__}, which is not callable; a builder returns "
                f"something the op can call with the tensors it was described"
            )
        if self._tune_requested:
            self._tune_target_kernel(device, kernel)
        return kernel

    def _tune_target_kernel(self, device: "torch.device | None", kernel: object) -> None:
        """Put a kernel this op's target built for *device* in tuned mode, on that device.

        A kernel that defines no ``request_tune`` cannot be tuned, and the op warns once.
        """
        tune = getattr(kernel, "request_tune", None)
        if not callable(tune):
            self._warn_untunable_target_kernel()
            return
        with device_context(device):
            tune()

    def _manifest_params(self) -> dict[str, object]:
        """The op's manifest params, by name, with the values this instance holds.

        ``build_kernel`` is called with these by keyword. Names come from the manifest
        (``_params_codegen``), values off the instance, so a param the manifest defaults to
        null arrives as the instance holds it: ``None`` unless the caller gave a value.

        Raises:
            AttributeError: The op declares a manifest param it keeps under another name.
                The manifest is the contract, so the op is what changes.
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
        """Return a read-only view of the entries built for *interface* so far, whoever built them.

        Empty before the interface's first build. A target serves the whole op, so for an op a
        target serves every interface shows the target's kernels, one per input signature. For
        introspection — tests, benchmark reporting — never for dispatch: an execution path
        asks ``kernel_for`` so a miss builds rather than raises.
        """
        if self._has_target_builder():
            return MappingProxyType(self._target_kernels)
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
                specialization (its params and construction tensors). Under ``None`` a sub-op
                does not see a param edited in place (assign a new value), a write to an
                inference construction tensor, or, in a traced call, a construction tensor
                replaced since the last eager call.
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
            calls = _OPEN_CALLS.__dict__.get("stack")
            frame = calls[-1] if calls else None
            plan = frame.plan if frame is not None and frame.op is self else None
            tensors = self._construction_tensors_key()
            if plan is None:
                return self._settled_delegate(stage, params, tensors)
            # By request position: one plan's forward makes the same requests in order.
            held, index = plan.delegates, frame.delegate_index
            frame.delegate_index = index + 1
            if index < len(held):
                entry = held[index]
                if entry[0] == stage and entry[2] == tensors:
                    return entry[1]
            with plan.filling:
                delegate = self._settled_delegate(stage, params, tensors)
                del held[index:]
                held.append((stage, delegate, tensors))
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
                    # Which stage holds each sub-op.
                    self._delegate_stages[id(delegate)] = stage
                    # A sub-op held while this op is in tuned mode joins it, built or given.
                    if self._tune_requested:
                        delegate.request_tune()
        return entries[identity]

    def _construction_tensors_key(self) -> "tuple | None":
        """Each construction-time tensor by identity and version; ``None`` in a trace."""
        if torch.compiler.is_compiling():
            return None
        signature = getattr(type(self), "_signature", None)
        names = signature.ctor_names if signature is not None else ()
        if not names:
            return ()
        tensors = (getattr(self, name, None) for name in names)
        return tuple(
            None if t is None else (id(t), None if t.is_inference() else t._version)
            for t in tensors
        )

    def _settled_delegate(self, stage: str, params: dict, tensors: "tuple | None") -> "Op":
        """The sub-op of *stage* for this op's params and construction tensors (*tensors*),
        built from *params* on a miss; one built for other tensors is replaced."""
        signature = getattr(type(self), "_signature", None)
        identity = (stage, None if signature is None else param_state(self, signature.sig))
        if tensors is not None:
            built_for = self.__dict__.setdefault("_settled_tensors", {})
            entries = self._delegates.setdefault(stage, {})
            with self._building:
                held = entries.get(identity)
                if held is not None and built_for.get(identity) != tensors:
                    # Replaced: no plan may reach it again.
                    for plan in self._plans.values():
                        at = next((i for i, e in enumerate(plan.delegates) if e[1] is held), None)
                        if at is not None:
                            del plan.delegates[at:]
                    del entries[identity]
                    # A call still running on the replaced sub-op records under its stage.
                    retired = self.__dict__.setdefault("_retired_delegates", {})
                    older = retired.pop(identity, None)
                    if older is not None:
                        del self._delegate_stages[id(older)]
                    retired[identity] = held
                delegate = self.delegate_for(stage, identity, **params)
                built_for[identity] = tensors
            return delegate
        return self.delegate_for(stage, identity, **params)

    def _held_delegates(self) -> Sequence["Op"]:
        """Return the sub-ops this op holds, in stage order, then in the order they were built.

        Derived from what :meth:`delegate_for` holds; an op does not override it.
        """
        held = self._delegates
        return tuple(op for stage in self.delegate_types for op in held.get(stage, {}).values())

    @property
    def serving_target(self) -> Target:
        """The target that serves this instance, selected by the first call that runs one.

        ``None`` until a call selects it, and again if the call that selected it fails. ``BUILTIN``
        for the in-tree implementation however it was chosen — ``target=BUILTIN``, the
        process default, or no target claiming the device. Otherwise the target's name.
        """
        if self._target_builder is _UNRESOLVED:
            return None
        if self._target_builder is None:
            return BUILTIN
        return self._serving_target

    def kernel_config(self) -> Optional[dict]:
        """The configuration the op's kernels were built with, or ``None``.

        An op given a config of its own answers with it; otherwise the first configured
        kernel does, which speaks for the whole call: one ``iter_kernels`` yields, else one
        a target built, whose ``config`` is optional.
        """
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
        """Yield every ``Kernel`` instance the op's entries hold, each one once.

        Reached: the entries of every interface, and the same walk over each
        ``_held_delegates()`` entry. A kernel on any other attribute is not
        searched for — an op that holds one builds it through an interface.

        What ``request_tune`` tunes and ``kernel_config`` reads of the in-tree kernels. A
        target's kernels are not ``Kernel`` instances and are not yielded here: those two
        reach them by device through ``_target_kernels_by_device``. ``_entries_built`` shows
        every entry, whoever built it.
        """
        seen: set[int] = set()
        for op in self._walk_ops():
            held = [entry for entries in op._built_entries.values() for entry in entries.values()]
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
            for signature, kernel in op._target_kernels.items():
                device = signature[0]
                if (device, id(kernel)) not in seen:
                    seen.add((device, id(kernel)))
                    yield op, device, kernel

    def _device_without_call_tensors(self) -> "torch.device | None":
        """The call device of a call that carries no tensor: the declared ``device`` param,
        else the construction-time tensors' device, else the accelerator's current one.

        The same rule the generated checks apply, for a caller that runs no check: a traced
        call, or an op body building a call spec for an op with no tensor input.
        """
        from tileops._ops._signature_codegen import tensorless_device

        sig = type(self)._signature.sig
        held = tuple(
            tensor
            for name, decl in sig.ctor_tensors.items()
            if not decl.cpu and (tensor := getattr(self, name, None)) is not None
        )
        return tensorless_device(type(self).__name__, self, held)

    def _declared_device(self) -> "torch.device | None":
        """Where the call runs when no tensor says: the ``device`` param an op declares.

        Only an op with no tensor input declares one, and ``None`` there leaves the
        choice to the target.
        """
        if "device" not in getattr(self, PARAM_NAMES_ATTRIBUTE, ()):
            return None
        device = getattr(self, "device", None)
        return None if device is None else torch.device(device)

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
        stack: list["Op"] = [self]
        while stack:
            op = stack.pop()
            if id(op) in seen:
                continue
            seen.add(id(op))
            yield op
            stack.extend(reversed(op._held_delegates()))

    def request_tune(self) -> None:
        """Put the op in tuned mode: what it holds now, and what it builds next.

        It applies to specializations that do not exist yet — an op tuned before its
        first fp16 call is tuned when bf16 arrives later — because ``_tune_requested`` is what
        carries it: every entry built while it is set has its kernels put in tuned mode, so
        no kernel factory needs to read it. A sub-op receives it from ``delegate_for``.
        Tuning a kernel is idempotent; one without ``autotune_configs`` stays untuned.

        A target's kernels are reached through the kernel, never the builder: each call of
        this calls ``request_tune()`` on each one built so far, once for each device it was
        built for, with that device current, and each one built later receives it before
        its first call. An op a
        target serves warns once when one of its kernels defines no ``request_tune()``.

        A kernel whose tuning raises leaves the op in tuned mode, so calling this again
        tunes what was not reached.
        """
        for op in self._walk_ops():
            op._tune_requested = True
        for kernel in self.iter_kernels():
            kernel.request_tune()
        for op, device, kernel in self._target_kernels_by_device():
            op._tune_target_kernel(device, kernel)

    def _warn_untunable_target_kernel(self) -> None:
        """Warn, once per instance, that a kernel of this op's target cannot be tuned."""
        if self._tune_warned:
            return
        self._tune_warned = True
        warnings.warn(
            f"{type(self).__name__} is served by target {self._serving_target!r}, whose "
            f"kernel defines no request_tune",
            UserWarning,
            stacklevel=4,
        )

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Run the op."""
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Make the op callable.

        ``forward`` is the op's computation, and it runs inside :meth:`_execute`. An op with
        a compile boundary calls the generated ``_call_boundary``, which runs :meth:`_execute`
        directly or as its operator's eager body. An op without one binds the call to
        ``forward``'s signature and calls :meth:`_execute` itself. Traced, an op without a
        compile boundary runs no check: an instance a target serves calls the target's kernel,
        and any other runs ``forward``. A traced first call binds the target as an eager one does.

        Raises:
            TypeError: The op has no manifest entry.
        """
        if self.compile_op_names:
            return self._call_boundary(*args, **kwargs)
        if type(self)._signature is None:
            raise self._no_entry()
        if torch.compiler.is_compiling():
            if self._target_builder is _UNRESOLVED:
                self._bind_target(self._traced_device((*args, *kwargs.values())))
            if self._has_target_builder():
                inputs, writes, execution = self._split_forward_args(args, kwargs)
                return self._run_on_target(inputs, writes, None, execution)
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
        self.__dict__.pop("_target_built", None)

    def _bind_target(self, device: "torch.device | None") -> None:
        """Decide which target serves this instance on the call *device*, and remember its
        builder.

        Once decided it does not change: the kernels this instance has built belong to that
        target. An instance is therefore bound to that target's devices — handing it tensors
        from elsewhere is a caller error, and the kernel is what reports it. *device* is the
        one the signature check placed the call on; a call with none probes nothing and
        decides nothing.

        A composite — an op that builds no kernel of its own — needs no builder: without
        one its sub-ops each select a target of their own.

        Raises:
            OpNotAvailableError: The selected target registers no builder for this op,
                and the op builds kernels of its own.
        """
        target = select_target(self.target, device)
        if target is None:
            self._serving_target = None
            if device is not None:
                self._target_builder = None  # a device was probed, so the answer is decided
            return
        if target is BUILTIN:
            self._serving_target = BUILTIN
            self._target_builder = None
            return
        # A target serves only the ops of tileops, which it knows by their manifest keys.
        builder = (
            registered_kernel_builder(type(self).__name__, target) if in_tree(type(self)) else None
        )
        if builder is None and not self.kernel_types:
            self._serving_target = target
            self._target_builder = None
            return
        if builder is None:
            raise OpNotAvailableError(
                f"target {target!r} registers no kernel builder for "
                f"{type(self).__name__}; targets that do: "
                f"{registered_targets(type(self).__name__)}. There is no fall back to the "
                f"in-tree implementation: those kernels do not run on this target's "
                f"devices."
            )
        self._serving_target = target
        self._target_builder = builder
