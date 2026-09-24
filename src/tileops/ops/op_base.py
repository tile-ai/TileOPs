import contextlib
import dataclasses
import functools
import inspect
import math
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
from tileops.manifest import WORKSPACE_ATTR, forward_signature, load_manifest
from tileops.manifest.rule_eval import bind_declared_shapes, eval_shape_rule

from ._output_dtype import output_dtype
from .compile_boundary import register_instance

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()

_Entry = TypeVar("_Entry")


class _Unresolved:
    """The type of :data:`_UNRESOLVED`, so a traceback says what it is."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<not resolved yet>"


# ``Op._builder`` before the first call. Distinct from ``None``, the decided answer
# "run the in-tree implementation".
_UNRESOLVED = _Unresolved()


@functools.lru_cache(maxsize=1)
def _declared_dispatch_keys() -> frozenset[str]:
    """Every dispatch key the manifest declares, across all ops.

    A key outside this set names no op's kernel anywhere: a typo, or a name that
    was renamed out of existence. A key inside it may still be unknown to the op
    being constructed, because a composite hands each sub-op the whole set the
    caller gave it and one sub-op's key is another's stranger.

    An empty set disables the check: a manifest that cannot be read says nothing about
    which keys exist, and refusing every override on that basis would stop ops that are
    otherwise fine from constructing.
    """
    keys: set[str] = set()
    try:
        entries = load_manifest().values()
    except Exception:  # noqa: BLE001 - an unreadable manifest disables the check, not the op
        return frozenset()
    for entry in entries:
        source = entry.get("source") if isinstance(entry, dict) else None
        declared = source.get("kernel_map") if isinstance(source, dict) else None
        if isinstance(declared, dict):
            keys.update(declared)
    return frozenset(keys)


_RECORDING_CALLS = False


@contextlib.contextmanager
def _recording_roofline_calls() -> "Iterator[None]":
    """Have every op call inside this block remember its input shapes and dtypes.

    ``eval_roofline_read_bytes()`` prices the write half from the output
    shapes, which the input shapes decide, and an op keeps only what its own
    ``eval_roofline`` needs. The NCU bytes audit wraps the call it reads that
    declaration off.

    Instrumentation, not operator interface, and off outside the block: the
    recording costs about a microsecond per call, which every benchmark row
    would otherwise carry.
    """
    global _RECORDING_CALLS
    previous = _RECORDING_CALLS
    _RECORDING_CALLS = True
    try:
        yield
    finally:
        _RECORDING_CALLS = previous


@functools.lru_cache(maxsize=None)
def _forward_input_names(op_name: str) -> tuple[str, ...]:
    """The op's ``forward`` input names, or empty when the manifest has none.

    Cached per op: every call records its tensors, and reading the manifest
    each time costs more than the rest of the recording together.
    """
    entry = load_manifest().get(op_name)
    if entry is None:
        return ()
    try:
        return tuple(forward_signature(entry)["inputs"])
    except Exception:
        return ()


class Op(ABC):
    """Base class for TileOPs operations.

    Attributes:
        kernel: single kernel, for ops that hold one; ops that build per
            specialization use ``kernel_for`` instead
        dtype: Data type for computation (e.g., torch.float16)
        input_shapes: Expected input tensor shapes

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
    dtype: Optional[torch.dtype] = None
    # This call's input shapes and dtypes, while a recording block is open.
    _roofline_call_tensors: Optional[dict] = None
    input_shapes: Optional[list[tuple]] = None
    # Whether kernels this op builds tune themselves. A ctor kwarg on the ops
    # that offer one, and what ``autotune()`` sets; a factory reads it when it
    # runs, so it governs every build that follows.
    tune: bool = False

    # Set of (input_index, axis) pairs identifying static (ctor-committed) axes.
    # `input_index` is the position in *input_shapes; `axis` is a non-negative
    # axis index within that shape. Subclasses set this to reflect their
    # manifest `static_dims`. Default empty = no committed axes.
    _static_axes: frozenset[tuple[int, int]] = frozenset()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-install manifest-derived methods on concrete subclasses.

        Synthesizes ``_validate_dtypes`` (per docs/design/ops-design.md
        §Step 5) and ``eval_roofline`` (per docs/design/roofline.md §4.4)
        from the subclass's manifest entry, attaches the manifest param names a
        backend's ``build_kernel`` is called with, and registers the compile-boundary
        operators the subclass declares. Each codegen pass is a no-op
        when the subclass does not advertise manifest metadata, supplies
        its own override, or is marked ``status: spec-only``.
        """
        super().__init_subclass__(**kwargs)
        from tileops.ops._compile_boundary_codegen import maybe_install_compile_boundary
        from tileops.ops._dtype_codegen import maybe_install_validator
        from tileops.ops._params_codegen import maybe_install_param_names
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline

        maybe_install_validator(cls)
        maybe_install_eval_roofline(cls)
        maybe_install_param_names(cls)
        maybe_install_compile_boundary(cls)

    @property
    @abstractmethod
    def default_kernel_map(self) -> dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    # Operators this op registers on the torch.compile boundary. Naming them is what lets
    # a test assert the traced graph holds nothing else, which is what keeps the graph the
    # same when another target serves the op. A tuple because a conditional in-place write
    # registers two. Registration happens once per class, so this is class state; an op
    # that declares ``torch_compile_fullgraph`` names its operators, which
    # ``register_compile_contract`` requires.
    compile_op_names: ClassVar[tuple[str, ...]] = ()

    # One ``OperatorSpec`` per operator the op registers; ``_compile_boundary_codegen``
    # turns them into the registrations and fills in ``compile_op_names``. Empty leaves
    # the op off the boundary.
    compile_boundary: ClassVar[tuple[object, ...]] = ()

    @abstractmethod
    def _infer_output_shapes(self, **shape_kwargs: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Infer output tensor shapes from input shapes.

        Concrete ops override this with a signature matching the named input
        shapes declared in their manifest ``shape_rules`` section (e.g.
        ``_infer_output_shapes(self, x_shape, weight_shape)``). The uniform
        ``**shape_kwargs`` base signature exists only to make the L1 contract
        grepable and discoverable; see docs/design/ops-design.md §``_infer_output_shapes``.
        Abstract: a concrete op supplies the body, and the validator's C6 check names
        it when a class inherits this one instead.
        """
        raise NotImplementedError(
            "_infer_output_shapes must be implemented by the concrete Op subclass; "
            "see docs/design/ops-design.md §`_infer_output_shapes` (codegen)"
        )

    @abstractmethod
    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        """Validate dtypes of input tensors passed to ``forward``.

        Concrete ops override this with a signature matching their manifest
        ``signature.inputs`` (e.g. ``_validate_dtypes(self, x, weight)``).
        See docs/design/ops-design.md §``_validate_dtypes``.
        """
        raise NotImplementedError(
            "_validate_dtypes must be implemented by the concrete Op subclass; "
            "see docs/design/ops-design.md §`_validate_dtypes` (codegen)"
        )

    @abstractmethod
    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Per docs/design/roofline.md §4.4 and §4.4.6, each concrete op's
        ``eval_roofline`` body is emitted by codegen as plain Python directly
        over ``self.*`` attributes — there is no shared roofline expression
        evaluator at L1, by design (§4.4.6 rejects "Op-local AST evaluator").
        The L1 base only declares the contract; concrete ops supply the body.
        """
        raise NotImplementedError(
            "eval_roofline must be implemented by the concrete Op subclass, "
            "emitted per docs/design/roofline.md §4.4 (codegen); the L1 base "
            "intentionally does not provide a generic evaluator — see "
            "docs/design/roofline.md §4.4.6 (Evaluator Surface Boundary)"
        )

    def eval_roofline_read_bytes(self) -> Optional[int]:
        """The read half of ``eval_roofline()[1]``, for the NCU bytes audit.

        ``bytes`` minus the write half, which the signature settles: every
        declared output once, plus a ``mutated`` input that is not an output.
        An op that reads only part of an input needs no override -- its
        ``bytes`` already counted that part.

        Returns:
            The read half in bytes, or ``None`` when the call has not bound what
            the write half needs.
        """
        write_bytes = self._roofline_write_bytes()
        if write_bytes is None:
            return None
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

    def _roofline_write_bytes(self) -> Optional[int]:
        """Bytes this call writes, from the signature alone, or ``None`` when the
        call has not bound the shapes or dtypes that price them."""
        from tileops.manifest import load_manifest
        from tileops.ops._output_dtype import output_dtype

        entry = load_manifest().get(type(self).__name__)
        if entry is None:
            return None
        signature = entry.get("signature") or {}
        inputs = signature.get("inputs") or {}
        outputs = signature.get("outputs") or {}
        order = list(inputs)
        recorded = getattr(self, "_roofline_call_tensors", None) or {}
        shapes = []
        for name in order:
            if name in recorded:
                shapes.append(recorded[name][0])
                continue
            bound = getattr(self, name, None)
            shape = getattr(bound, "shape", None) or getattr(self, f"{name}_shape", None)
            shapes.append(None if shape is None else tuple(shape))
        try:
            out_shapes = self._infer_output_shapes(*shapes)
        except Exception:
            return None
        dtype = getattr(self, "dtype", None)
        total = 0
        for name, shape in out_shapes.items():
            try:
                elem = output_dtype(self, name, dtype).itemsize
            except Exception:
                return None
            total += math.prod(shape) * elem
        # A ``mutated`` input is written too, unless that write is the output's:
        # an op with an ``inplace`` param may write into the input it read.
        has_inplace = "inplace" in (signature.get("params") or {})
        for name, spec in inputs.items():
            if not (spec or {}).get("mutated") or name in outputs or has_inplace:
                continue
            shape = shapes[order.index(name)]
            if shape is None:
                continue
            if name in recorded:
                elem = recorded[name][1].itemsize
            else:
                bound = getattr(self, name, None)
                elem = getattr(getattr(bound, "dtype", None), "itemsize", None)
            if elem is None:
                return None
            total += math.prod(shape) * elem
        return total

    def compute_roof(self) -> str:
        """GPU-profile key of the compute unit that prices this op's FLOPs.

        ``eval_roofline()`` counts the work; ``compute_roof()`` names the
        peak that bounds it (docs/design/roofline.md §1.2). The key is a
        statement about the *optimal* implementation, declared by the op
        author — never inferred from the running kernel, so a kernel on the
        wrong unit is still measured against the right ceiling.

        The base default covers ops whose arithmetic runs on CUDA cores in
        fp32 (elementwise, reductions, norms, scans). An op whose FLOPs are
        matmul contractions overrides this with ``tensor_core_roof(self.dtype)``
        (or a backend-specific key). Valid whenever ``eval_roofline()`` is —
        after the dtype is bound.
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
        declared = _declared_dispatch_keys()
        if not declared:
            return
        stale = sorted(set(override) - declared - set(own))
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

    def forwarded_overrides(self) -> Optional[dict[str, Kernel]]:
        """The caller's replacements, to hand to a sub-op this op builds.

        Only what the caller supplied. A composite op that passed its whole
        resolved ``kernel_map`` down would mark every key as replaced, and a
        replacement that cannot serve a call is an error rather than something to
        select around.
        """
        if not self._overridden_keys or not self.kernel_map:
            return None
        return {
            key: cls for key, cls in self.kernel_map.items() if key in self._overridden_keys
        } or None

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
        self._refuse_empty_input(inputs)

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
            entries[key] = build()
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
            OpNotAvailableError: What :meth:`_get_or_build_kernel` raises.
        """
        return self._get_or_build_kernel(role, inputs, lambda: self.entry_for(role, call))

    @classmethod
    @functools.cache
    def _forward_io(cls) -> "tuple[tuple[str, ...], frozenset[str]]":
        """The ``forward`` inputs a target is called with, and which of them it writes.

        The names are ``signature.inputs`` followed by ``resources.workspaces``; an op the
        manifest does not describe has none. The written ones are the inputs marked
        ``mutated`` and every workspace, which is scratch the kernel writes.
        """
        entry = load_manifest().get(cls.__name__)
        inputs = forward_signature(entry)["inputs"] if entry is not None else {}
        mutated = frozenset(
            name
            for name, attrs in inputs.items()
            if isinstance(attrs, dict) and (attrs.get("mutated") or attrs.get(WORKSPACE_ATTR))
        )
        return tuple(inputs), mutated

    @classmethod
    @functools.cache
    def _forward_parameters(cls) -> inspect.Signature:
        """``forward``'s signature, read once per class: a target call binds it every time."""
        return inspect.signature(cls.forward)

    @classmethod
    @functools.cache
    def _forward_outputs(cls) -> "tuple[str, ...]":
        """The op's declared outputs, in order."""
        entry = load_manifest().get(cls.__name__)
        return tuple((entry or {}).get("signature", {}).get("outputs") or ())

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
        writes = {
            name: value
            for name, value in bound.arguments.items()
            if name not in names and isinstance(value, torch.Tensor)
        }
        return inputs, writes

    def _served_by_target(self) -> bool:
        """Whether a target's builder, rather than the in-tree kernels, serves this instance."""
        return self._builder is not None and self._builder is not _UNRESOLVED

    def _serve(
        self,
        *inputs: "torch.Tensor | None",
        _written: "frozenset[str] | None" = None,
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
        if settled_here:
            # ``__call__`` settled this already — unless it was traced. Dynamo defers a
            # traced frame's attribute writes until after the graph has run, so the
            # operator body arrives here still ``_UNRESOLVED``.
            self._resolve_builder(inputs, writes)
        try:
            if self._served_by_target():
                return self._call_target(inputs, writes, _written)
            return self._eager_forward(*inputs, **writes)
        except Exception:
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
    ) -> object:
        """Run the whole op on the target this instance settled on.

        What the op layer guarantees every target: every tensor on one device, every
        input the call does not write contiguous, and — checked once per signature — the
        input dtypes and shape rules the manifest states. The kernel is built once per
        device and per input dtype and shape. A call that completes leaves
        ``self.<input>_shape`` and ``self.dtype``, the state the op's roofline reads.

        Raises:
            ValueError: The call breaks that guarantee, or every output would be empty.
            OpNotAvailableError: The builder returned something that is not callable.
        """
        devices = {t.device for t in (*inputs, *writes.values()) if t is not None}
        if len(devices) > 1:
            raise ValueError(
                f"{type(self).__name__} needs every tensor on one device; got "
                f"{sorted(map(str, devices))}"
            )
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
        # A caller's output buffer is checked too, but the builder never sees it.
        checked = signature + tuple(
            (name, t.dtype, tuple(t.shape)) for name, t in sorted(writes.items())
        )
        seen = getattr(self, "_target_checked", None)
        if seen is None:
            seen = self._target_checked = set()
        if checked not in seen:
            self._check_target_call(named, writes)
            seen.add(checked)
        kernels = getattr(self, "_target_kernels", None)
        if kernels is None:
            kernels = self._target_kernels = {}
        kernel = kernels.get(signature)
        if kernel is None:
            kernel = kernels[signature] = self._build_target_kernel(named)
        result = kernel(*inputs, **writes)
        for name, t in zip(names, inputs, strict=True):
            setattr(self, f"{name}_shape", None if t is None else tuple(t.shape))
        self.dtype = next((t.dtype for t in inputs if t is not None), self.dtype)
        # An op whose every output is an input it writes returns nothing, as in tree.
        outputs = self._forward_outputs()
        if outputs and all(name in named for name in outputs):
            return None
        return result

    def _check_target_call(
        self, inputs: "dict[str, torch.Tensor | None]", writes: "dict[str, torch.Tensor]"
    ) -> None:
        """Hold a call the target has not seen yet to the manifest.

        A caller-supplied output buffer is the op's output, so it is held to that
        output's dtype and shape rules under the output's name.

        Raises:
            ValueError: An input dtype, the buffer's dtype, or a shape rule the manifest
                states does not hold, or every output would be empty.
        """
        check = getattr(type(self), "_validate_manifest_dtypes", None)
        if check is not None:
            check(self, **{name: t for name, t in inputs.items() if t is not None})
        signature = (load_manifest().get(type(self).__name__) or {}).get("signature") or {}
        outputs = tuple(signature.get("outputs") or ())
        filled = {}
        if writes and len(outputs) == 1:
            (buffer,) = writes.values()
            dtype = next((t.dtype for t in inputs.values() if t is not None), None)
            expected = output_dtype(self, outputs[0], dtype)
            if buffer.dtype != expected:
                raise ValueError(
                    f"{type(self).__name__}: the output buffer is {buffer.dtype}, but "
                    f"{outputs[0]!r} is {expected}"
                )
            filled = {outputs[0]: buffer}
        try:
            extents = bind_declared_shapes(signature, inputs)
        except ValueError as exc:
            raise ValueError(f"{type(self).__name__}: {exc}") from None
        scope = {**extents, **inputs, **filled, **self._manifest_params()}
        for rule in signature.get("shape_rules") or ():
            holds, unevaluable = eval_shape_rule(rule, scope)
            if not holds and unevaluable is None:
                raise ValueError(
                    f"{type(self).__name__}: this call breaks the manifest shape rule {rule!r}"
                )
        self._refuse_empty_input(tuple(inputs.values()))

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

    def kernel_delegates(self) -> Sequence["Op"]:
        """Return the ops whose kernels this op runs.

        A composite op — one that resolves its call through another op rather
        than building the kernel itself — overrides this so enumeration reaches
        the delegate. Default: this op builds everything it runs.
        """
        return ()

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
            stack.extend(op.kernel_delegates())

    def autotune(self) -> None:
        """Put the op in tuned mode: what it holds now, and what it builds next.

        It applies to specializations that do not exist yet — an op tuned before its
        first fp16 call is tuned when bf16 arrives later — because ``tune`` is what
        carries it, and a kernel factory reads that flag when it runs.

        A target's builder is not passed ``tune``, so the flag cannot reach what a target
        builds; an op a target serves warns once instead of ignoring the request.
        """
        for op in self._walk_ops():
            op.tune = True
            if op.settled_target not in (None, BUILTIN):
                op._warn_tune_not_passed()
        for kernel in self.iter_kernels():
            kernel.autotune()

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
        settled_here = self._builder is _UNRESOLVED
        if settled_here:
            self._resolve_builder(args, kwargs)
        try:
            if self._served_by_target() and not self.compile_op_names:
                result = self._call_target(*self._bind_forward(args, kwargs))
            else:
                result = self.forward(*args, **kwargs)
        except Exception:
            if settled_here:
                self._unsettle()
            raise
        if _RECORDING_CALLS or self._roofline_call_tensors is not None:
            self._track_roofline_call(args, kwargs)
        return result

    def _track_roofline_call(self, args: tuple, kwargs: dict) -> None:
        """Keep the record of this call's input tensors current.

        Outside a recording block, and under ``torch.compile`` where building
        the dict would break the graph, the record is dropped rather than left
        describing an earlier call.
        """
        if not _RECORDING_CALLS or torch.compiler.is_compiling():
            self._roofline_call_tensors = None
            return
        names = _forward_input_names(type(self).__name__)
        if not names:
            return
        # A call may omit an optional input, so the lists need not be equal.
        recorded = {}
        for name, value in zip(names, args, strict=False):
            if isinstance(value, torch.Tensor):
                recorded[name] = (tuple(value.shape), value.dtype)
        for name, value in kwargs.items():
            if name in names and isinstance(value, torch.Tensor):
                recorded[name] = (tuple(value.shape), value.dtype)
        self._roofline_call_tensors = recorded

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
        # A workspace is declared under ``resources`` but passed to forward() like
        # any other tensor, so it counts toward the argument list this compares.
        names = tuple(forward_signature(entry)["inputs"])
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
        """Undo a settling whose call did not finish, dropping what it built."""
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

    def _resolve_builder(self, args: tuple, kwargs: dict) -> None:
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
        device = self._first_tensor_device(args, kwargs) or self._declared_device()
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

    def _cache_key(self, *input_shapes: tuple[int, ...]) -> Hashable:
        """Return a cache key for kernel dispatch given forward-time input shapes.

        The default is every axis not named by ``self._static_axes``. Correct for any
        op, but with ``_static_axes`` empty it compiles once per distinct input shape
        and warns once per subclass.

        Override to project the shape onto whatever the kernel math depends on — for
        example, flattening leading dims to one product when the kernel treats the
        input as 2D.
        """
        if not self._static_axes and type(self)._cache_key is Op._cache_key:
            cls = type(self)
            if cls not in _EMPTY_STATIC_DIMS_WARNED:
                _EMPTY_STATIC_DIMS_WARNED.add(cls)
                warnings.warn(
                    f"{cls.__name__}: Op._cache_key() called with empty "
                    f"_static_axes and no subclass override. The default "
                    f"keys the kernel cache by the full input shape, which "
                    f"produces one compile per distinct shape under dynamic "
                    f"inputs. Override _cache_key to project onto whatever "
                    f"the kernel math actually depends on.",
                    UserWarning,
                    stacklevel=2,
                )
        return tuple(
            s
            for i, shape in enumerate(input_shapes)
            for axis, s in enumerate(shape)
            if (i, axis) not in self._static_axes
        )
