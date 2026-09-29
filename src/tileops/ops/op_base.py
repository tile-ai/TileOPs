import contextlib
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
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch

from tileops.backend import (
    BUILTIN,
    OpNotAvailableError,
    Target,
    TensorSpec,
    registered_targets,
)
from tileops.backend.dispatch import registered_kernel_builder, select_target
from tileops.backend.registry import IMPLEMENTATIONS, ensure_loaded
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface

from .compile_boundary import register_instance

_Entry = TypeVar("_Entry")


# Every dispatch key a created op class declares in ``kernel_types``. Constructing an op imports
# it and every sub-op it builds, so every key that can replace something in that op is here.
_DISPATCH_KEYS: set[str] = set()


class _Unresolved:
    """The type of :data:`_UNRESOLVED`, so a traceback says what it is."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<not resolved yet>"


# ``Op._builder`` before the first call. Distinct from ``None``, the decided answer
# "run the in-tree implementation".
_UNRESOLVED = _Unresolved()


# The calls in progress on this thread, innermost last, each with the checked calls completed
# inside it: what a composite's call collects from its sub-ops.
_OPEN_CALLS = threading.local()


def _open_calls() -> list:
    calls = getattr(_OPEN_CALLS, "stack", None)
    if calls is None:
        calls = _OPEN_CALLS.stack = []
    return calls


class Op(ABC):
    """Base class for TileOPs operations.

    Attributes:
        kernel: single kernel, for ops that hold one; ops that build per
            specialization use ``kernel_for`` instead
        dtype: Data type for computation (e.g., torch.float16)

    Properties:
        total_flops (optional): Total flops for the op.
            If specified, will be used to calculate TFlops in profile().
        total_memory (optional): Total memory for the op.
            If specified, will be used to calculate Bandwidth in profile().
    """

    # Which set of kernels serves this instance: a target name, ``BUILTIN`` for the in-tree
    # implementation, or None to decide from the input device. Constructor-only: it settles
    # kernel identity, so it must not vary per call.
    target: Target = None
    # The resolved answer: ``_UNRESOLVED``, ``None`` (in-tree), or a target's build_kernel.
    _builder: object = _UNRESOLVED
    # Which target that was, for introspection and error messages.
    _settled_target: Target = None
    # Whether this instance has warned that a tuning request cannot reach its target.
    _tune_warned: bool = False

    # An entry the op keeps bound directly, if it keeps one: a ``Kernel`` in-tree, whatever a
    # target's builder returned otherwise. Specializations are held per interface.
    kernel: Optional[Callable[..., object]]
    # The implementation that runs under each key, ``kernel_map=`` included: read-only once
    # installed.
    kernel_map: Optional[Mapping[str, Kernel]] = None
    # The implementation registered under each key, in-tree or by a backend. Its applicability
    # and precedence select the key, whatever ``kernel_map=`` put there to run.
    _registered: Mapping[str, type[Kernel]] = MappingProxyType({})
    # Each key mapped to the keys its registered implementation is preferred over, transitively.
    _preferred: Mapping[str, frozenset[str]] = MappingProxyType({})
    # Built entries, ``{interface: {key: entry}}``. Annotation only: the instance
    # attribute appears on the first ``kernel_for`` call, so an op that
    # has built nothing carries no dict, and no constructor declares one.
    _built_entries: dict[str, dict[Hashable, object]]
    # The keys of each kernel interface's implementations, as installed.
    _interface_keys: Mapping[str, tuple[str, ...]] = MappingProxyType({})
    # Resolved entries, ``{(interface, call spec): entry}``. Annotation only, like
    # ``_built_entries``.
    _dispatched: dict[tuple[str, Hashable], object]
    # The ``kernel_map=`` the caller passed, as given; what every sub-op this op builds is handed.
    _given_kernel_map: Optional[dict[str, Kernel]] = None
    # Held sub-ops, ``{stage: {key: op}}``. Annotation only, like ``_built_entries``.
    _delegates: dict[str, dict[Hashable, "Op"]]
    dtype: Optional[torch.dtype] = None
    # Whether kernels this op builds tune themselves. A ctor kwarg on the ops
    # that offer one, and what ``autotune()`` sets; a factory reads it when it
    # runs, so it governs every build that follows.
    tune: bool = False

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Install what the subclass's manifest entry generates, and the param names a
        backend's ``build_kernel`` is called with.

        A subclass without an entry gets neither. An entry's ``status`` does not change what
        its signature generates.
        """
        super().__init_subclass__(**kwargs)
        _DISPATCH_KEYS.update(cls.__dict__.get("kernel_types", {}))
        from tileops.ops._params_codegen import maybe_install_param_names
        from tileops.ops._signature_codegen import maybe_install_signature

        maybe_install_signature(cls)
        maybe_install_param_names(cls)

    # The op's dispatch keys and the kernel class each names. An op with no kernel of its own
    # (a composite) declares none.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = MappingProxyType({})

    # The places this op calls a kernel, each named and mapped to its kernel interface. An
    # interface's implementations are the keys whose registered class inherits it, in-tree or
    # added by a backend (``tileops.backend.register_implementation``).
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = MappingProxyType({})

    # The ops this op holds as sub-ops, by stage name in stage order: the sub-op counterpart of
    # ``kernel_types``. Every sub-op is held through ``delegate_for``.
    delegate_types: ClassVar[Mapping[str, type["Op"]]] = MappingProxyType({})

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        """This instance's dispatch table: every entry of ``kernel_types``, unless a
        construction parameter selects some of them."""
        return dict(self.kernel_types)

    @property
    def last_call(self) -> object:
        """The ``SignatureCall`` of this op's last completed call: its ``ix``, tensors and effects.

        Raises:
            RuntimeError: No call has completed yet.
        """
        call = getattr(self, "_signature_call", None)
        if call is None:
            raise RuntimeError(f"{type(self).__name__}: no call has completed yet")
        return call

    # Operators this op registers on the torch.compile boundary. Naming them is what lets
    # a test assert the traced graph holds nothing else, which is what keeps the graph the
    # same when another target serves the op. A tuple because a conditional in-place write
    # registers two. Registration happens once per class, so this is class state.
    compile_op_names: ClassVar[tuple[str, ...]] = ()

    # Whether this op declares a compile boundary, which is its claim that it supports
    # ``fullgraph=True``; the signature generates its operators.
    compile_boundary: ClassVar[bool] = False

    # Injected implementation objects ``__init__`` takes beyond ``signature.params`` and the
    # execution-policy parameters every op takes (docs/design/manifest.md § Signature).
    execution_parameters: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def _infer_output_shapes(self, *shapes: tuple[int, ...], dtypes=None) -> dict:
        """Infer output tensor shapes from input shapes, in ``signature.inputs`` order.

        Generated from the op's manifest entry; abstract so that a class without one
        cannot be instantiated.
        """
        raise NotImplementedError("generated from the op's manifest entry")

    @abstractmethod
    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        """Validate the ``forward`` inputs against the signature.

        Generated from the op's manifest entry; abstract so that a class without one
        cannot be instantiated.
        """
        raise NotImplementedError("generated from the op's manifest entry")

    @abstractmethod
    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last completed call.

        Generated from the op's manifest ``roofline`` (docs/design/roofline.md §4.4).
        """
        raise NotImplementedError("generated from the op's manifest entry")

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

    def roofline_inputs(self) -> "dict[str, int]":
        """What decided this call's ``bytes``, where its inputs' values decided it.

        Two calls of one shape can move different amounts -- a routed MoE reads
        the experts its routing selected -- and the benchmark records this
        beside the reading so such a number says why it moved.

        Nothing judges it, and it is not part of ``(flops, bytes)``. Empty
        unless the op's traffic follows its inputs' values.
        """
        return {}

    def compute_roof(self) -> str:
        """GPU-profile key of the compute unit that prices this op's FLOPs.

        ``eval_roofline()`` counts the work; ``compute_roof()`` names the
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

    def _refuse_unknown_keys(self, override: dict[str, Kernel], own: "Iterable[str]") -> None:
        """Raise for a replacement under a name neither this op nor any other has.

        Such a name replaces nothing, and the call that follows runs the shipped
        implementation as if the caller had asked for it — which is what a key that was
        renamed looks like from the outside. A name some other op has is left alone: a
        composite hands every sub-op the whole set, so one sub-op's key reaches the rest.
        *own* covers an op the manifest does not describe, such as one a test declares.
        Only an op with a map of its own asks this: a composite stores what it is given
        verbatim and hands it down, and the sub-op that owns the name is the one that can
        tell a stale key from a sibling's.

        Raises:
            ValueError: *override* names a key nothing declares.
        """
        if not _DISPATCH_KEYS:
            return
        stale = sorted(set(override) - _DISPATCH_KEYS - set(own))
        if stale:
            raise ValueError(
                f"{type(self).__name__} was given kernel_map keys no op has: {stale}. "
                f"A key nothing declares replaces nothing, so the call would run the "
                f"shipped implementation. This op's keys: {sorted(own)}"
            )

    def _install_kernel_map(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Install the resolved kernel map onto ``self.kernel_map``, read-only.

        Each key's registered implementation, from ``default_kernel_map`` or a backend, is
        replaced by *kernel_map*'s under the same name; only what runs changes, not what
        selects the key. A name no op in the library declares is refused; a name this op does
        not have but another does is kept out of the resolved map and ignored, because that
        is how a composite's sub-ops see each other's keys. Resolving a kernel *class* needs
        no device, so construction does not probe one: an op constructs wherever it is
        imported, and a target that cannot run it surfaces when a kernel is first selected,
        built or called. Installing again drops everything the previous installation built.

        Raises:
            ValueError: A backend registered a key this op already has, or what
                :meth:`_refuse_unknown_keys` or :meth:`_install_interfaces` raises.
            TypeError: What :meth:`_install_interfaces` raises.
        """
        for built in ("_built_entries", "_selected_entries", "_dispatched"):
            self.__dict__.pop(built, None)
        default_map = self.default_kernel_map
        override = dict(kernel_map) if kernel_map else {}
        self._given_kernel_map = override or None
        if default_map is None or len(default_map) == 0:
            # Composite op: store override verbatim. Its keys belong to the sub-ops it
            # builds, which is where a name nothing declares is refused.
            self.kernel_map = MappingProxyType(override)
            return
        name = type(self).__name__
        added = IMPLEMENTATIONS.get(name, {})
        taken = sorted(set(added) & set(default_map))
        if taken:
            raise ValueError(f"implementations registered for {name} reuse keys it has: {taken}")
        registered = {**default_map, **added}
        if override:
            self._refuse_unknown_keys(override, registered)
        self._registered = MappingProxyType(registered)
        self.kernel_map = MappingProxyType(
            {key: override.get(key, cls) for key, cls in registered.items()}
        )
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
        self._preferred = MappingProxyType(preferred)
        if self.interfaces:
            self._interface_keys = self._install_interfaces()

    def _install_interfaces(self) -> Mapping[str, tuple[str, ...]]:
        """Return each kernel interface's keys, each implementation checked against it.

        A key belongs to every interface its registered implementation inherits. What runs
        under the key, ``kernel_map=`` included, must inherit the same interface, take its
        ``forward`` arguments, and state ``entry_for`` as a classmethod.

        Raises:
            TypeError: An implementation breaks its interface's contract.
            ValueError: A key belongs to no interface, an interface has two general
                implementations, or a ``preferred_over`` names no other implementation of
                its interface, is stated by the general one, or is cyclic.
        """
        name = type(self).__name__
        interface_keys = {}
        for where, interface in self.interfaces.items():
            keys = tuple(k for k, cls in self._registered.items() if issubclass(cls, interface))
            arguments = list(inspect.signature(interface.forward).parameters)[1:]
            for key in keys:
                runs = self.kernel_map[key]
                if not (issubclass(runs, Kernel) and issubclass(runs, interface)):
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__} does not implement "
                        f"{interface.__name__}"
                    )
                if not isinstance(inspect.getattr_static(runs, "entry_for"), classmethod):
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__}.entry_for is not a classmethod"
                    )
                try:
                    inspect.signature(runs.forward).bind(None, *arguments)
                except TypeError as exc:
                    raise TypeError(
                        f"{name}.{where} {key!r}: {runs.__name__}.forward does not take "
                        f"{interface.__name__}'s arguments: {exc}"
                    ) from None
            general = [k for k in keys if self._registered[k].general]
            if len(general) > 1:
                raise ValueError(
                    f"{name}.{where} has more than one general implementation: {general}"
                )
            for key in keys:
                preferred = self._registered[key].preferred_over
                if not preferred <= set(keys) - {key} or (preferred and key in general):
                    raise ValueError(
                        f"{name}.{where} {key!r} is preferred over {sorted(preferred)}; it names "
                        f"other implementations of its interface, and the general one names none"
                    )
                if key in self._preferred[key]:
                    raise ValueError(f"{name}.{where} preferences form a cycle through {key!r}")
            interface_keys[where] = keys
        unassigned = sorted(
            set(self._registered) - {k for ks in interface_keys.values() for k in ks}
        )
        if unassigned:
            raise ValueError(f"{name} keys implement none of its kernel interfaces: {unassigned}")
        return MappingProxyType(interface_keys)

    def select_kernel_key(self, keys: "tuple[str, ...]", call: object) -> str:
        """Return the one key among *keys* whose implementation serves *call*.

        A key is available where its registered implementation or the one ``kernel_map=``
        put there can run, and it applies where the registered one's ``refusal`` accepts the
        call. Among the available keys that apply, the answer is the unique one no other is
        preferred over: ``preferred_over`` orders them, transitively, and the general one is
        below every other. Neither the order of *keys* nor any numeric priority decides it.

        A key selected with a ``kernel_map=`` replacement is served by the replacement, and
        a replacement that cannot run or does not serve the call is an error rather than
        the registered implementation standing in for it.

        Raises:
            OpNotAvailableError: No key runs on the call's device type.
            ValueError: When no implementation serves the call, when the replacement
                behind the selected key cannot, or when several are preferred over none.
        """
        device = getattr(call, "device", None)
        on_device = device is None
        rules: dict[str, type[Kernel]] = {}
        rejected: list[str] = []
        for key in keys:
            runs = (self.kernel_map or {}).get(key)
            if runs is None:
                continue
            rule = self._registered.get(key, runs)
            on_device = on_device or device.type in rule.devices or device.type in runs.devices
            reason = rule.unavailable(call)
            if reason is not None and runs is not rule and runs.unavailable(call) is None:
                reason = None
            reason = reason or rule.refusal(call)
            if reason is None:
                rules[key] = rule
            else:
                rejected.append(f"{key} ({rule.__name__}: {reason})")
        chosen = [
            key
            for key, rule in rules.items()
            if not any(
                key in self._preferred.get(other, ()) or (rule.general and not rules[other].general)
                for other in rules
                if other != key
            )
        ]
        if len(chosen) == 1:
            (key,) = chosen
            runs = self.kernel_map[key]
            reason = None if runs is rules[key] else runs.unavailable(call) or runs.refusal(call)
            if reason is not None:
                raise ValueError(
                    f"the kernel supplied for {key} through kernel_map= ({runs.__name__}) "
                    f"cannot serve this call: {reason}. Call: {call}"
                )
            return key
        if not on_device and rejected:
            raise OpNotAvailableError(
                f"{type(self).__name__}'s in-tree kernels do not run on {device}; known "
                f"targets for this op: {registered_targets(type(self).__name__)}"
            )
        if not chosen:
            raise ValueError(
                "no implementation serves this call: "
                + "; ".join(rejected or ["no implementation is installed"])
                + f". Call: {call}"
            )
        raise ValueError(
            f"dispatch is ambiguous: {', '.join(chosen)} all serve this call, and none is "
            f"preferred over the others. Overlapping implementations state which one wins "
            f"through ``preferred_over``, and an interface has at most one general one. "
            f"Call: {call}"
        )

    def select_implementation(self, interface: str, call: CallSpec) -> str:
        """Return the key of the implementation of *interface* that serves *call*.

        Raises:
            OpNotAvailableError: What :meth:`select_kernel_key` raises.
            ValueError: What :meth:`select_kernel_key` raises.
        """
        return self.select_kernel_key(self._interface_keys[interface], call)

    # FIXME(staged-rollout): selection among every installed key, across kernel interfaces.
    #
    # Broken invariant: an interface's implementations are the keys that inherit it
    #     (ops-design.md § Kernel selection).
    # Why: unmigrated ops and their tests select through this without declaring interfaces.
    # Cleanup: when every op declares ``interfaces``, delete this and select through
    #     ``select_implementation``.
    def select_kernel(self, call: object, keys: "tuple[str, ...] | None" = None) -> type[Kernel]:
        """Return the implementation that serves *call*.

        A name is the handle ``kernel_map=`` replaces an implementation by; the class
        is the answer. *keys* defaults to every key installed.

        Raises:
            ValueError: What :meth:`select_kernel_key` raises.
        """
        return self.kernel_map[self.select_kernel_key(keys or tuple(self.kernel_map or ()), call)]

    def dispatch_kernel(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map (auto-discovery entry point)."""
        ensure_loaded()  # before any traced region, which the first call may be inside
        check = getattr(type(self), "_check_construction", None)
        if check is not None:
            check(self)
        self._install_kernel_map(kernel_map)
        self._instance_key = register_instance(self)

    def _get_or_build_kernel(
        self,
        name: str,
        inputs: "Sequence[torch.Tensor | None]",
        plan: Callable[[], Entry],
    ) -> _Entry:
        """Return the in-tree entry for this call, building it once on a miss.

        The memoization primitive under :meth:`kernel_for`, which is what an op calls.

        Args:
            name: Which of this op's kernels is being asked for.
            inputs: The tensors this kernel will be handed. An ``optional: true`` input the
                call did not pass occupies its slot as ``None``.
            plan: The in-tree identity and builder.

        Returns:
            The stored entry, identical across calls describing the same specialization.

        Raises:
            OpNotAvailableError: The op has no in-tree implementation for *name*.
        """

        # Plain attribute reads and dict lookups, no ``self.__dict__``: this
        # runs inside a dynamo-traced forward on every cache hit, and dynamo
        # cannot trace a method call on an instance ``__dict__``.
        roles = getattr(self, "_built_entries", None)
        if roles is None:
            roles = {}
            self._built_entries = roles
        entries = roles.get(name)
        if entries is None:
            entries = {}
            roles[name] = entries

        key, build = plan()
        if build is None:
            raise OpNotAvailableError(
                f"{type(self).__name__} has no in-tree implementation for {name!r}, "
                f"so it needs a target that registers one; known targets for this "
                f"op: {registered_targets(type(self).__name__)}"
            )
        # A build reads the device it is built on (its architecture, SM count, compile
        # target), so it is built with the call's device current: its tensors' device, or
        # for an op with no tensor input the device it declares.
        device = next((t.device for t in inputs if t is not None and t.is_cuda), None)
        if device is None:
            declared = self._declared_device()
            device = declared if declared is not None and declared.type == "cuda" else None
        if key not in entries:
            with torch.cuda.device(device) if device is not None else contextlib.nullcontext():
                entry = build()
            if self.tune:
                for kernel in self._entry_kernels(entry):
                    kernel.request_tune()
            entries[key] = entry
        return entries[key]

    # FIXME(staged-rollout): the memoization an op without kernel interfaces reaches the cache by.
    #
    # Broken invariant: every kernel is reached through a kernel interface, one lookup per hit
    #     (ops-design.md § Kernel selection).
    # Why: unmigrated ops override this, or select among every key through it, per call.
    # Cleanup: when every op declares ``interfaces``, delete this, ``_get_or_build_kernel``
    #     and the branch of ``kernel_for`` for a role that is no interface.
    def entry_for(self, role: str, call: object) -> Entry:
        """How to build what serves *call* for *role*, and what keys the result.

        The default asks the implementation every installed key selects for *call*. An op
        with one implementation and no call record overrides this and states its own
        identity and builder. An op with nothing in tree has no builder, and the caller
        reports that.

        Raises:
            ValueError: What :meth:`select_kernel` raises.
        """
        if not self.kernel_map:
            return None, None
        # A repeated call record is a lookup: selection and the identity depend on
        # nothing else.
        selected = getattr(self, "_selected_entries", None)
        if selected is None:
            selected = {}
            self._selected_entries = selected
        entry = selected.get((role, call))
        if entry is None:
            cls = self.select_kernel(call)
            identity, build = cls.entry_for(call)
            entry = ((cls, identity), build)
            selected[(role, call)] = entry
        return entry

    def kernel_for(
        self,
        interface: str,
        inputs: "Sequence[torch.Tensor | None]",
        call: object = None,
    ) -> object:
        """Return the in-tree entry that serves *call* for *interface*, resolving it on a miss.

        The one way an op's in-tree implementation reaches a kernel. It runs only when the
        in-tree kernels serve the op: a target serves the whole op instead
        (:meth:`_call_target`). For a kernel interface a hit is one lookup of
        ``(interface, call)``; a miss selects the implementation and builds or reuses the entry
        its build identity names (:meth:`_resolve_entry`).

        Args:
            interface: The kernel interface, one of ``interfaces``. An op that declares none
                names the memoization bucket of one kernel it runs, never the name of an
                implementation it chose.
            inputs: The tensors this kernel will be handed; an op that declares no interfaces
                has its device checked from them.
            call: The call spec: for an interface, an instance of its ``request``.

        Raises:
            ValueError: What :meth:`_resolve_entry` or :meth:`entry_for` raises.
            TypeError: What :meth:`_resolve_entry` raises.
            OpNotAvailableError: No implementation runs on the call's device, or what
                :meth:`_get_or_build_kernel` raises.
        """
        if interface in self.interfaces:
            dispatched = getattr(self, "_dispatched", None)
            if dispatched is None:
                dispatched = {}
                self._dispatched = dispatched
            try:
                entry = dispatched.get((interface, call))
            except TypeError:
                # The miss path's checks name what makes the call spec unusable.
                self._resolve_entry(interface, call)
                raise
            if entry is None:
                entry = self._resolve_entry(interface, call)
                dispatched[(interface, call)] = entry
            return entry
        self._refuse_device(inputs, call)
        return self._get_or_build_kernel(interface, inputs, lambda: self.entry_for(interface, call))

    def _resolve_entry(self, interface: str, call: CallSpec) -> object:
        """Resolve the entry serving *call* for *interface*, on a miss of the dispatch cache.

        The device facts are read from the call's device here, by selection and the
        builder, and the builder runs with that device current. Two call specs whose
        implementation names one build identity share one entry. Tuning acts on the
        resolved entry, never through the builder.

        Raises:
            TypeError: *call* is not the interface's ``request`` type, holds a field that
                cannot key the cache, or states a device fact.
            ValueError: What :meth:`select_implementation` raises.
            OpNotAvailableError: What :meth:`select_implementation` raises.
        """
        request = self.interfaces[interface].request
        if not isinstance(call, request):
            raise TypeError(
                f"{type(self).__name__}.{interface} takes a {request.__name__} call spec, "
                f"not {type(call).__name__}"
            )
        call.refuse_unkeyable()
        if call.stated_device_facts:
            raise TypeError(
                f"{type(self).__name__}.{interface} reads the device facts from the call's "
                f"device; this call spec states {sorted(call.stated_device_facts)}"
            )
        cls = self.kernel_map[self.select_implementation(interface, call)]
        identity, build = cls.entry_for(call)
        roles = getattr(self, "_built_entries", None)
        if roles is None:
            roles = {}
            self._built_entries = roles
        entries = roles.setdefault(interface, {})
        entry = entries.get((cls, identity))
        if entry is None:
            on_cuda = call.device is not None and call.device.type == "cuda"
            with torch.cuda.device(call.device) if on_cuda else contextlib.nullcontext():
                entry = build()
            entries[(cls, identity)] = entry
        if self.tune:
            for kernel in self._entry_kernels(entry):
                kernel.request_tune()
        return entry

    def _refuse_device(self, inputs: "Sequence[torch.Tensor | None]", call: object) -> None:
        """Raise when no implementation in the kernel map declares the call's device type.

        Each implementation states the devices it runs on (``Kernel.devices``), so a replacement
        that runs elsewhere is asked about its own. The call's device is the call record's, else
        its inputs' (a CPU-resident input yields to any other), else the op's ``device`` parameter.
        """
        tensors = sorted((t for t in inputs if t is not None), key=lambda t: t.device.type == "cpu")
        device = getattr(call, "device", None) or (tensors[0].device if tensors else None)
        device = torch.device(device) if device is not None else self._declared_device()
        classes = (self.kernel_map or {}).values()
        if device is None or not classes:
            return
        if not any(device.type in getattr(c, "devices", Kernel.devices) for c in classes):
            raise OpNotAvailableError(
                f"{type(self).__name__}'s in-tree kernels do not run on {device}; known targets "
                f"for this op: {registered_targets(type(self).__name__)}"
            )

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

    def _bind_forward(self, args: tuple, kwargs: dict) -> "tuple[tuple, dict[str, torch.Tensor]]":
        """Split a ``forward`` call into its manifest inputs and its written buffers.

        The inputs come back in manifest order, an absent optional one as ``None``. A
        tensor ``forward`` takes beyond them is a caller-supplied output buffer, such as
        ``out``.
        """
        bound = self._forward_parameters().bind(self, *args, **kwargs)
        bound.apply_defaults()
        names, _ = self._forward_io()
        inputs = tuple(bound.arguments.get(name) for name in names)
        # The check reports a non-tensor `out` by name.
        writes = {
            name: value
            for name, value in bound.arguments.items()
            if name not in names
            and (isinstance(value, torch.Tensor) or (name == "out" and value is not None))
        }
        return inputs, writes

    def _check_signature(
        self, inputs: "tuple[torch.Tensor | None, ...]", writes: "dict[str, torch.Tensor]"
    ) -> object:
        """Run the checks generated from the entry's signature."""
        plan = type(self)._signature
        self._open_call()
        return plan.check(self, {**dict(zip(plan.sig.inputs, inputs, strict=True)), **writes})

    def _complete_signature(
        self, call: object, result: object, inputs: tuple, writes: "dict[str, torch.Tensor]"
    ) -> None:
        """Hold what the implementation returned to the checked call, then keep the call.

        The kept call is what ``eval_roofline`` prices, so only a completed eager one is kept.
        """
        if call is None:
            return
        from tileops.ops._signature_codegen import check_result

        sig = type(self)._signature.sig
        check_result(
            sig,
            call,
            result,
            {**dict(zip(sig.inputs, inputs, strict=True)), **writes},
            tuple(getattr(self, t, None) for t in sig.ctor_tensors),
        )
        if torch.compiler.is_compiling():
            self._drop_call()
        else:
            self._keep_call(call)

    def _open_call(self) -> None:
        """Start collecting the checked calls this op's sub-ops complete during its call."""
        _open_calls().append((self, []))

    def _drop_call(self) -> None:
        """Close a call that did not complete; nothing it collected is kept."""
        calls = _open_calls()
        if calls and calls[-1][0] is self:
            calls.pop()

    def _keep_call(self, call: object) -> None:
        """Keep *call* as the last completed one, with the checked calls its sub-ops completed
        during it, by stage and in completion order (docs/design/roofline.md §2.2), and report
        it to the call this one ran inside."""
        calls = _open_calls()
        collected = calls.pop()[1] if calls and calls[-1][0] is self else []
        held = getattr(self, "_delegates", None) or {}
        stage_of = {id(op): stage for stage, ops in held.items() for op in ops.values()}
        stages = {stage: [] for stage in self.delegate_types}
        for op, done in collected:
            if id(op) in stage_of:
                stages[stage_of[id(op)]].append(done)
        call = dataclasses.replace(call, stages={k: tuple(v) for k, v in stages.items()})
        self._signature_call = call
        if calls:
            calls[-1][1].append((self, call))

    def _execution_arguments(self, args: tuple, kwargs: dict) -> "dict[str, object]":
        """What ``forward`` takes after the signature's inputs and ``out``, bound by name."""
        bound = self._forward_parameters().bind(self, *args, **kwargs)
        bound.apply_defaults()
        prefix = {"self", "out", *self._forward_io()[0]}
        return {n: v for n, v in bound.arguments.items() if n not in prefix}

    def _served_by_target(self) -> bool:
        """Whether a target's builder, rather than the in-tree kernels, serves this instance."""
        return self._builder is not None and self._builder is not _UNRESOLVED

    def _serve(
        self,
        *inputs: "torch.Tensor | None",
        _written: "frozenset[str] | None" = None,
        _execution: "dict[str, object] | None" = None,
        **writes: torch.Tensor,
    ) -> object:
        """Run one call on whichever set of kernels serves this instance.

        The body of every compile-boundary operator, which is handed exactly the manifest
        inputs. The in-tree kernels run ``_eager_forward``; a target runs the whole op.
        *_written* names the inputs this operator's kernel writes, when that is not every
        input the manifest marks ``mutated``: an op that registers an inplace companion
        writes nothing through its default operator.

        Raises:
            OpNotAvailableError: What :meth:`_resolve_builder` raises.
        """
        settled_here = self._builder is _UNRESOLVED
        try:
            call = self._check_signature(inputs, writes)
            # An empty call runs no implementation, so none has to be available for it.
            empty = self._writes_nothing(call)
            if settled_here and not empty:
                self._resolve_builder(inputs, writes, call.device)
            if empty:
                result = self._empty_result(call, inputs, writes)
            elif self._served_by_target():
                result = self._call_target(inputs, writes, _written, _execution)
            else:
                result = self._eager_forward(*inputs, **writes, **(_execution or {}))
            self._complete_signature(call, result, inputs, writes)
            return result
        except Exception:
            self._drop_call()
            # Whoever settled it unsettles it. ``__call__``'s handler does not run when
            # the failure comes out of a compiled graph, so this one has to.
            if settled_here:
                self._unsettle()
            raise

    def _call_target(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        writes: "dict[str, torch.Tensor]",
        written: "frozenset[str] | None" = None,
        execution: "dict[str, object] | None" = None,
    ) -> object:
        """Run the whole op on the target this instance settled on.

        What the op layer guarantees every target: every tensor on one device, every
        input the call does not write contiguous, and the checks generated from the signature.
        The kernel is built once per device and per input dtype and shape.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        devices = {t.device for t in (*inputs, *writes.values()) if t is not None}
        # The generated checks placed the call, `device: cpu` tensors aside.
        devices = {d for d in devices if d.type != "cpu"} or devices
        names, mutated = self._forward_io()
        written = mutated if written is None else written
        inputs = tuple(
            t if t is None or name in written else t.contiguous()
            for name, t in zip(names, inputs, strict=True)
        )
        device = next(iter(devices)) if devices else self._declared_device()
        # An absent optional input keeps its slot as ``None``, so two calls differing
        # only in which one they omit key apart.
        signature = (device,) + tuple(
            None if t is None else (t.dtype, tuple(t.shape)) for t in inputs
        )
        named = dict(zip(names, inputs, strict=True))
        kernels = getattr(self, "_target_kernels", None)
        if kernels is None:
            kernels = self._target_kernels = {}
        kernel = kernels.get(signature)
        if kernel is None:
            kernel = kernels[signature] = self._build_target_kernel(named)
        result = kernel(*inputs, **writes, **(execution or {}))
        # An op whose every output is an input it writes returns nothing, as its forward does.
        outputs = self._forward_outputs()
        if outputs and all(name in named for name in outputs):
            return None
        return result

    def _build_target_kernel(self, inputs: "dict[str, torch.Tensor | None]") -> object:
        """Ask the target for the kernel serving calls described by *inputs*.

        Raises:
            OpNotAvailableError: The builder returned something that is not callable.
        """
        if self.tune:
            self._warn_tune_not_passed()
        params = self._manifest_params()
        specs = tuple(None if t is None else TensorSpec.of(t) for t in inputs.values())
        kernel = self._builder(*specs, **params)
        if not callable(kernel):
            raise OpNotAvailableError(
                f"target {self._settled_target!r} built {kernel!r} for "
                f"{type(self).__name__}, which is not callable; a builder returns "
                f"something the op can call with the tensors it was described"
            )
        return kernel

    def _manifest_params(self) -> dict[str, object]:
        """The op's manifest params, by name, with the values this instance settled on.

        ``build_kernel`` is called with these by keyword. Names come from the manifest
        (``_params_codegen``), values off the instance, so a param the manifest defaults to
        null arrives as the instance holds it: ``None`` unless the caller gave a value.

        Raises:
            AttributeError: The op declares a manifest param it keeps under another name.
                The manifest is the contract, so the op is what changes.
        """
        names = getattr(self, "__manifest_param_names__", None)
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

    def built_kernels(self, interface: str) -> Mapping[Hashable, object]:
        """Return a read-only view of the entries built for *interface* so far, whoever built them.

        Empty before the interface's first build. A target serves the whole op, so for an op a
        target serves every interface shows the target's kernels, one per input signature. For
        introspection — tests, benchmark reporting — never for dispatch: an execution path
        asks ``kernel_for`` so a miss builds rather than raises.
        """
        if self._served_by_target():
            return MappingProxyType(getattr(self, "_target_kernels", None) or {})
        roles = getattr(self, "_built_entries", None) or {}
        return MappingProxyType(roles.get(interface, {}))

    def delegate_for(
        self, stage: str, key: Hashable, given: "Op | None" = None, /, **params: object
    ) -> "Op":
        """Return the sub-op held for *stage* under *key*, building it on a miss.

        The one way an op holds a sub-op. A miss builds ``delegate_types[stage](**params)``
        with this op's ``target``, the caller's ``kernel_map`` as given and this op's current
        ``tune``; *given*, an implementation the caller injected for the stage, is held as is
        instead. Eager only, like :meth:`kernel_for`: a traced ``forward`` reaches no miss.

        Args:
            stage: A key of ``delegate_types``.
            key: The sub-op's identity: everything that can change what gets built.
            given: The caller's implementation for this stage, or ``None`` to build one.
            params: The sub-op's constructor arguments other than the execution policy.

        Raises:
            KeyError: *stage* is not declared in ``delegate_types``.
        """
        cls = self.delegate_types[stage]
        held = getattr(self, "_delegates", None)
        if held is None:
            held = {}
            self._delegates = held
        entries = held.setdefault(stage, {})
        if key not in entries:
            entries[key] = (
                given
                if given is not None
                else cls(
                    **params, target=self.target, kernel_map=self._given_kernel_map, tune=self.tune
                )
            )
        return entries[key]

    def kernel_delegates(self) -> Sequence["Op"]:
        """Return the sub-ops this op holds, in stage order, then in the order they were built.

        Derived from what :meth:`delegate_for` holds; an op does not override it.
        """
        held = getattr(self, "_delegates", None) or {}
        return tuple(op for stage in self.delegate_types for op in held.get(stage, {}).values())

    @property
    def settled_target(self) -> Target:
        """Which implementation a call settled this instance on.

        ``None`` until a call settles it, and again if that settling call fails. ``BUILTIN``
        for the in-tree implementation however it was chosen — ``target=BUILTIN``, the
        process default, or no target claiming the device. Otherwise the target's name.
        """
        if self._builder is _UNRESOLVED:
            return None
        if self._builder is None:
            return BUILTIN
        return self._settled_target

    def run_config(self) -> Optional[dict]:
        """The configuration the op's kernels were built with, or ``None``.

        An op given a config of its own answers with it; otherwise the first
        configured kernel ``iter_kernels`` yields does, which speaks for the whole call.
        """
        own = getattr(self, "config", None)
        if own:
            return own
        for kernel in self.iter_kernels():
            config = getattr(kernel, "config", None)
            if config:
                return config
        return None

    def iter_kernels(self) -> Iterator[Kernel]:
        """Yield every ``Kernel`` instance the op's entries hold, each one once.

        Reached: the entries of every interface, ``self.kernel``, and the same walk over
        each ``kernel_delegates()`` entry. A kernel on any other attribute is not
        searched for — an op that holds one builds it through an interface.

        What ``autotune`` tunes and ``run_config`` reads. An entry holding no ``Kernel``
        — a target builder's plain callable — contributes nothing here;
        ``built_kernels`` shows every entry, whoever built it.
        """
        seen: set[int] = set()
        for op in self._walk_ops():
            roles = getattr(op, "_built_entries", None) or {}
            held = [entry for entries in roles.values() for entry in entries.values()]
            held.append(getattr(op, "kernel", None))
            for entry in held:
                for kernel in self._entry_kernels(entry):
                    if id(kernel) not in seen:
                        seen.add(id(kernel))
                        yield kernel

    def _declared_device(self) -> "torch.device | None":
        """Where the call runs when no tensor says: the ``device`` param an op declares.

        Only an op with no tensor input declares one, and ``None`` there leaves the
        choice to the target.
        """
        if "device" not in getattr(self, "__manifest_param_names__", ()):
            return None
        device = getattr(self, "device", None)
        return None if device is None else torch.device(device)

    @staticmethod
    def _first_tensor_device(args: tuple, kwargs: dict) -> "torch.device | None":
        """The device of the first tensor a call carries, one level into sequences."""
        for value in (*args, *kwargs.values()):
            if isinstance(value, torch.Tensor):
                return value.device
            if isinstance(value, (tuple, list)):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        return item.device
        return None

    @staticmethod
    def _entry_kernels(entry: object) -> "list[Kernel]":
        """Return the kernels one entry holds.

        An entry is a kernel, a sequence of kernels built together, or a dataclass
        carrying them alongside what else the specialization implies. An entry that
        hides its kernels from this walk is invisible to ``autotune``.
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
            stack.extend(reversed(op.kernel_delegates()))

    def autotune(self) -> None:
        """Put the op in tuned mode: what it holds now, and what it builds next.

        It applies to specializations that do not exist yet — an op tuned before its
        first fp16 call is tuned when bf16 arrives later — because ``tune`` is what
        carries it: every entry built while it is set has its kernels put in tuned mode, so
        no kernel factory needs to read it. A sub-op receives it from ``delegate_for``.
        Tuning a kernel is idempotent; one without ``autotune_configs`` stays untuned.

        A target's builder is not passed ``tune``, so the flag cannot reach what a target
        builds; an op a target serves warns once instead of ignoring the request.
        """
        for op in self._walk_ops():
            op.tune = True
            if op.settled_target not in (None, BUILTIN):
                op._warn_tune_not_passed()
        for kernel in self.iter_kernels():
            kernel.request_tune()

    def _warn_tune_not_passed(self) -> None:
        """Warn, once per instance, that tuning does not reach this op's target."""
        if self._tune_warned:
            return
        self._tune_warned = True
        warnings.warn(
            f"{type(self).__name__} is served by target {self._settled_target!r}, whose "
            f"build_kernel is not passed tune",
            UserWarning,
            stacklevel=3,
        )

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Run the op."""
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Make the op callable.

        Settles which set of kernels serves this instance, once. The in-tree kernels
        run ``forward``; a target runs the whole op. An op on the compile boundary
        branches inside its operator instead (:meth:`_serve`), so ``forward`` only picks
        which operator to call.

        A call that fails settles nothing, so one invalid call cannot aim the instance
        for good.
        """
        settled_here = self._builder is _UNRESOLVED and not self.compile_op_names
        try:
            call, bound = None, None
            # An op without a compile boundary claims no traced contract.
            if not self.compile_op_names and not torch.compiler.is_compiling():
                bound = self._bind_forward(args, kwargs)
                call = self._check_signature(*bound)
                if settled_here and not self._writes_nothing(call):
                    # The generated checks decide the call device, `device: cpu` tensors aside.
                    self._resolve_builder(args, kwargs, call.device)
            if call is not None and self._writes_nothing(call):
                result = self._empty_result(call, *bound)
            elif self._served_by_target() and not self.compile_op_names:
                bound = bound or self._bind_forward(args, kwargs)
                written = call.written if call is not None else None
                execution = self._execution_arguments(args, kwargs) if call is not None else None
                result = self._call_target(*bound, written, execution)
            else:
                result = self.forward(*args, **kwargs)
            if call is not None:
                self._complete_signature(call, result, *bound)
        except Exception:
            self._drop_call()
            if settled_here:
                self._unsettle()
            raise
        return result

    @staticmethod
    def _writes_nothing(call: object) -> bool:
        """Whether every tensor *call* writes, its outputs and written inputs, holds no elements.

        Such a call has nothing to compute, so no implementation runs it. The written tensors
        decide, not the inputs: an empty input whose output still holds elements, such as a
        sum over an empty axis, runs as any other call does.
        """
        written = [t for t, _, writes in call.traffic if writes]
        return bool(written) and not any(math.prod(call.tensors[t][0]) for t in written)

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
            elif decl.buffer and call.out:
                values.append(writes["out"])
            else:
                shape, dtype = call.tensors[name]
                values.append(torch.empty(shape, dtype=getattr(torch, dtype), device=call.device))
        return None if not values else values[0] if len(values) == 1 else tuple(values)

    def _unsettle(self) -> None:
        """Undo a settling whose call did not finish, dropping what it built.

        The sub-ops are unsettled with it, so none keeps what the failed call settled.
        """
        for delegate in self.kernel_delegates():
            delegate._unsettle()
        dropped = {
            id(entry)
            for entries in (getattr(self, "_built_entries", None) or {}).values()
            for entry in entries.values()
        }
        if id(getattr(self, "kernel", None)) in dropped:
            self.kernel = None
        self._builder = _UNRESOLVED
        self._settled_target = None
        self._built_entries = {}
        self._selected_entries = {}
        self._dispatched = {}
        self._target_kernels = {}
        self._target_checked = set()

    def _resolve_builder(
        self, args: tuple, kwargs: dict, device: "torch.device | None" = None
    ) -> None:
        """Decide which target serves this instance and remember its builder.

        Once decided it does not change: the kernels this instance has built belong to that
        target. An instance is therefore bound to that target's devices — handing it tensors
        from elsewhere is a caller error, and the kernel is what reports it. The device is
        the first tensor's, or else the op's ``device`` param; a call with neither probes
        nothing and decides nothing.

        A composite — an op that builds no kernel of its own — needs no builder: without
        one its sub-ops each settle on a target of their own.

        Raises:
            OpNotAvailableError: The selected target registers no builder for this op,
                and the op builds kernels of its own.
        """
        device = device or self._first_tensor_device(args, kwargs) or self._declared_device()
        target = select_target(self.target, device)
        if target is None:
            self._settled_target = None
            if device is not None:
                self._builder = None  # a device was probed, so the answer is decided
            return
        if target is BUILTIN:
            self._settled_target = BUILTIN
            self._builder = None
            return
        builder = registered_kernel_builder(type(self).__name__, target)
        if builder is None and not self.default_kernel_map:
            self._settled_target = target
            self._builder = None
            return
        if builder is None:
            raise OpNotAvailableError(
                f"target {target!r} registers no kernel builder for "
                f"{type(self).__name__}; targets that do: "
                f"{registered_targets(type(self).__name__)}. There is no fall back to the "
                f"in-tree implementation: those kernels do not run on this target's "
                f"devices."
            )
        self._settled_target = target
        self._builder = builder
