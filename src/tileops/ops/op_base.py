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
from tileops.backend.registry import ensure_loaded
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.manifest import load_manifest

from .compile_boundary import register_instance

_Entry = TypeVar("_Entry")


class _Unresolved:
    """The type of :data:`_UNRESOLVED`, so a traceback says what it is."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<not resolved yet>"


# ``Op._builder`` before the first call. Distinct from ``None``, the decided answer
# "run the in-tree implementation".
_UNRESOLVED = _Unresolved()


# Every dispatch key a created op class declares in ``kernel_types``. Constructing an op imports
# it and every sub-op it builds, so every key that can replace something in that op is here.
_DISPATCH_KEYS: set[str] = set()

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
    # target's builder returned otherwise. Specializations are held per role.
    kernel: Optional[Callable[..., object]]
    kernel_map: Optional[dict[str, Kernel]] = None
    # Built entries, ``{role: {key: entry}}``. Annotation only: the instance
    # attribute appears on the first ``kernel_for`` call, so an op that
    # has built nothing carries no dict, and no constructor declares one.
    _kernel_roles: dict[str, dict[Hashable, object]]
    # Dispatch keys the caller replaced through ``kernel_map=``.
    _overridden_keys: frozenset = frozenset()
    # The ``kernel_map=`` the caller passed, as given; what every sub-op this op builds is handed.
    _given_kernel_map: Optional[dict[str, Kernel]] = None
    # Held sub-ops, ``{stage: {key: op}}``. Annotation only, like ``_kernel_roles``.
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

    def _install_kernel_map(self, candidate_map: Optional[dict[str, Kernel]] = None) -> None:
        """Install the resolved kernel map onto ``self.kernel_map``.

        An entry of ``default_kernel_map`` is replaced by *candidate_map*'s under the
        same name. A name no op in the library declares is refused; a name this op does
        not have but another does is kept out of the resolved map and ignored, because
        that is how a composite's sub-ops see each other's keys. Resolving a
        kernel *class* needs no device, so construction does not probe one: an op
        constructs wherever it is imported, and a target that cannot run it surfaces when
        a kernel is first selected, built or called.

        Raises:
            ValueError: What :meth:`_refuse_unknown_keys` raises.
        """
        default_map = self.default_kernel_map
        override = dict(candidate_map) if candidate_map else {}
        self._given_kernel_map = override or None
        if default_map is None or len(default_map) == 0:
            # Composite op: store override verbatim. Its keys belong to the sub-ops it
            # builds, which is where a name nothing declares is refused.
            self.kernel_map = override
            self._overridden_keys = frozenset(override)
            return
        if override:
            self._refuse_unknown_keys(override, default_map)
        resolved: dict[str, Kernel] = {}
        for name, default_kernel in default_map.items():
            resolved[name] = override.get(name, default_kernel)
        self.kernel_map = resolved
        # Read by select_kernel_key: a replacement is never skipped silently.
        self._overridden_keys = frozenset(override) & frozenset(resolved)

    def select_kernel_key(self, keys: "tuple[str, ...]", call: object) -> str:
        """Return the one key among *keys* whose implementation serves *call*.

        Each candidate answers for itself: a specialised implementation states the
        region it serves, the one marked ``general`` runs where none of them does.
        Neither the order of *keys* nor any implementation naming another decides it.

        A replacement installed through ``kernel_map=`` is asked the same question as
        the class it replaced. A replacement that cannot serve the call is an error
        rather than a reason to fall back to the shipped implementation.

        Raises:
            ValueError: When no implementation serves the call, when a
                replacement cannot and a shipped one would stand in for it, or
                when two implementations both claim it.
        """
        applicable: list[str] = []
        rejected: list[str] = []
        refused_overrides: list[str] = []
        for key in keys:
            kernel_cls = (self.kernel_map or {}).get(key)
            if kernel_cls is None:
                continue
            reason = kernel_cls.refusal(call)
            if reason is None:
                applicable.append(key)
                continue
            rejected.append(f"{key} ({kernel_cls.__name__}: {reason})")
            if key in self._overridden_keys:
                refused_overrides.append(f"{key} ({kernel_cls.__name__}: {reason})")

        specialised = [k for k in applicable if not self.kernel_map[k].general]
        chosen = specialised or applicable

        if len(chosen) == 1:
            if refused_overrides and chosen[0] not in self._overridden_keys:
                raise ValueError(
                    "the kernel supplied for "
                    + "; ".join(refused_overrides)
                    + f" — selection does not fall back to the shipped '{chosen[0]}' "
                    f"when a replacement is in force. Call: {call}"
                )
            return chosen[0]
        if not chosen:
            lead = (
                "the kernel supplied for " + "; ".join(refused_overrides) + ", and "
                if refused_overrides
                else ""
            )
            raise ValueError(
                lead
                + "no implementation serves this call: "
                + "; ".join(rejected or ["no implementation is installed"])
                + f". Call: {call}"
            )
        raise ValueError(
            f"dispatch is ambiguous: {', '.join(chosen)} all serve this call, so none "
            f"is the answer. Implementations of one key must serve disjoint regions, "
            f"and at most one of them may be general. Call: {call}"
        )

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
        roles = getattr(self, "_kernel_roles", None)
        if roles is None:
            roles = {}
            self._kernel_roles = roles
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
        if key not in entries:
            entry = build()
            if self.tune:
                for kernel in self._entry_kernels(entry):
                    kernel.request_tune()
            entries[key] = entry
        return entries[key]

    def entry_for(self, role: str, call: object) -> Entry:
        """How to build what serves *call* for *role*, and what keys the result.

        The default asks the implementation this op's candidates select for *call*,
        which is where an op with more than one implementation stops. An op with one
        implementation and no call record overrides this and states its own identity
        and builder, so that every op reaches the cache through one path.

        An op with nothing in tree has no builder, and the caller reports that.

        Raises:
            ValueError: What :meth:`select_kernel` raises.
        """
        if not self.kernel_map:
            return None, None
        cls = self.select_kernel(call)
        identity, build = cls.entry_for(call)
        return (cls, identity), build

    def kernel_for(
        self,
        role: str,
        inputs: "Sequence[torch.Tensor | None]",
        call: object = None,
    ) -> object:
        """Return the in-tree kernel that serves *call* for *role*, building it on a miss.

        The one way an op's in-tree implementation reaches a kernel. It runs only when the
        in-tree kernels serve the op: a target serves the whole op instead
        (:meth:`_call_target`).

        Args:
            role: Which of this op's kernels is being asked for. One name per kernel
                the op runs, never the name of an implementation it chose.
            inputs: The tensors this kernel will be handed, for the empty-input guard.
            call: What describes this call, handed to :meth:`entry_for`. An op with
                nothing in tree states none.

        Raises:
            ValueError: What :meth:`entry_for` raises.
            OpNotAvailableError: No implementation this op holds runs on the call's device,
                or what :meth:`_get_or_build_kernel` raises.
        """
        self._refuse_empty_input(inputs)
        self._refuse_device(inputs, call)
        return self._get_or_build_kernel(role, inputs, lambda: self.entry_for(role, call))

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
            if settled_here:
                self._resolve_builder(inputs, writes, call.device)
            if self._served_by_target():
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
        null arrives as the number the op chose.

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

    def built_kernels(self, role: str) -> Mapping[Hashable, object]:
        """Return a read-only view of the entries built for *role* so far, whoever built them.

        Empty before the role's first build. A target serves the whole op, so for an op a
        target serves every role shows the target's kernels, one per input signature. For
        introspection — tests, benchmark reporting — never for dispatch: an execution path
        asks ``kernel_for`` so a miss builds rather than raises.
        """
        if self._served_by_target():
            return MappingProxyType(getattr(self, "_target_kernels", None) or {})
        roles = getattr(self, "_kernel_roles", None) or {}
        return MappingProxyType(roles.get(role, {}))

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

        Reached: the entries of every role, ``self.kernel``, and the same walk over
        each ``kernel_delegates()`` entry. A kernel on any other attribute is not
        searched for — an op that holds one builds it through a role.

        What ``autotune`` tunes and ``run_config`` reads. An entry holding no ``Kernel``
        — a target builder's plain callable — contributes nothing here;
        ``built_kernels`` shows every entry, whoever built it.
        """
        seen: set[int] = set()
        for op in self._walk_ops():
            roles = getattr(op, "_kernel_roles", None) or {}
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
                if settled_here:
                    # The generated checks decide the call device, `device: cpu` tensors aside.
                    self._resolve_builder(args, kwargs, call.device)
            if self._served_by_target() and not self.compile_op_names:
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

    def _refuse_empty_input(self, inputs: "Sequence[torch.Tensor | None]") -> None:
        """Raise for a call whose every declared output would hold no elements.

        Such a call leaves the launch a zero-sized grid, which reports itself as an
        internal assertion saying nothing about what is unsupported.

        The output decides, not the input: a zero-length axis on an input is legitimate
        wherever the op still produces something.

        Raises:
            ValueError: The op cannot produce the empty output this call asks for.
        """
        if not any(t is not None and t.numel() == 0 for t in inputs):
            return

        entry = load_manifest().get(type(self).__name__)
        if entry is None:
            return
        names = tuple(entry["signature"]["inputs"])
        if len(names) != len(inputs):
            return
        try:
            shapes = self._infer_output_shapes(
                *(None if t is None else tuple(t.shape) for t in inputs)
            )
        except (TypeError, ValueError, KeyError):
            return  # an op whose shape inference this order does not describe
        declared = entry["signature"]["outputs"]
        if any(name not in shapes for name in declared):
            return
        if any(math.prod(shapes[name]) for name in declared):
            return

        name, tensor = next(
            (n, t) for n, t in zip(names, inputs, strict=True) if t is not None and t.numel() == 0
        )
        raise ValueError(
            f"{type(self).__name__} does not support an empty tensor: input {name!r} has "
            f"shape {tuple(tensor.shape)}, which holds no elements."
        )

    def _unsettle(self) -> None:
        """Undo a settling whose call did not finish, dropping what it built.

        The sub-ops are unsettled with it, so none keeps what the failed call settled.
        """
        for delegate in self.kernel_delegates():
            delegate._unsettle()
        dropped = {
            id(entry)
            for entries in (getattr(self, "_kernel_roles", None) or {}).values()
            for entry in entries.values()
        }
        if id(getattr(self, "kernel", None)) in dropped:
            self.kernel = None
        self._builder = _UNRESOLVED
        self._settled_target = None
        self._kernel_roles = {}
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
