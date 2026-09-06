import dataclasses
import math
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
    TypeVar,
    Union,
)

import torch

from tileops.backend import (
    BUILTIN,
    BuildKernel,
    OpNotAvailableError,
    Target,
    TensorSpec,
    registered_targets,
)
from tileops.backend.dispatch import registered_kernel_builder, select_target
from tileops.backend.registry import ensure_loaded
from tileops.kernels.kernel_base import Kernel
from tileops.manifest import load_manifest

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


class Op(ABC):
    """Base class for TileOPs operations.

    A Op represents a computational operation with:
    - Hardware-aware kernel dispatch
    - Correctness testing via reference implementation
    - Performance profiling
    - Autotuning interface

    Attributes:
        kernel: single kernel, for ops that hold one; ops that build per
            specialization use ``get_or_build_kernel`` instead
        dtype: Data type for computation (e.g., torch.float16)
        device: Device for computation (e.g., 'cuda')
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

    kernel: Kernel
    kernel_map: Optional[dict[str, Kernel]] = None
    # Built entries, ``{role: {key: entry}}``. Annotation only: the instance
    # attribute appears on the first ``get_or_build_kernel`` call, so an op that
    # has built nothing carries no dict, and no constructor declares one.
    _kernel_roles: dict[str, dict[Hashable, object]]
    # Dispatch keys the caller replaced through ``kernel_map=``.
    _overridden_keys: frozenset = frozenset()
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = "cuda"
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
        from the subclass's manifest entry, and attaches the manifest param
        names a backend's ``build_kernel`` is called with. Each codegen pass is a no-op
        when the subclass does not advertise manifest metadata, supplies
        its own override, or is marked ``status: spec-only``. Codegen
        modules are lazy-imported to avoid a circular import at ``Op``
        definition time.
        """
        super().__init_subclass__(**kwargs)
        from tileops.ops._dtype_codegen import maybe_install_validator
        from tileops.ops._params_codegen import maybe_install_param_names
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline

        maybe_install_validator(cls)
        maybe_install_eval_roofline(cls)
        maybe_install_param_names(cls)

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

    def _install_kernel_map(self, candidate_map: Optional[dict[str, Kernel]] = None) -> None:
        """Install the resolved kernel map onto ``self.kernel_map``.

        Iterates ``self.default_kernel_map`` and, for each entry, picks the
        override from ``candidate_map`` when present, falling back to the
        default. Resolving a kernel *class* needs no device, so construction
        does not probe one: an op constructs wherever it is imported, and a
        target that cannot run the op surfaces when a kernel is first selected,
        built or called. Both auto-discovered and user-supplied maps share this
        single install path.
        """
        default_map = self.default_kernel_map
        override = dict(candidate_map) if candidate_map else {}
        if default_map is None or len(default_map) == 0:
            # Composite op: store override verbatim.
            self.kernel_map = override
            self._overridden_keys = frozenset(override)
            return
        resolved: dict[str, Kernel] = {}
        for name, default_kernel in default_map.items():
            resolved[name] = override.get(name, default_kernel)
        self.kernel_map = resolved
        # Which keys the caller replaced. A dispatch key served by several
        # implementations skips a default that cannot serve a call, but a
        # replacement the caller supplied is never skipped silently: the whole
        # point of the override is to run that implementation.
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

        The rule every family dispatches by. Each candidate answers for itself:
        a specialised implementation states the region it serves, and the one
        marked ``general`` runs where none of them does. Nothing is decided by
        the order the keys are written in, and no implementation names another.

        A replacement installed through ``kernel_map=`` is asked the same
        question as the class it replaced, so a specialisation can be swapped
        without the general implementation knowing. When a replacement cannot
        serve the call and a shipped implementation would take its place, that
        is an error: the caller supplied it so that it would run, and a result
        from the shipped kernel would be read as theirs.

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

    def dispatch_kernel(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map (auto-discovery entry point)."""
        ensure_loaded()  # before any traced region, which the first call may be inside
        self._install_kernel_map(kernel_map)
        # Conforming __init__s all pass through here — the zero-boilerplate
        # registration point for the compile dispatch boundary.
        self._instance_key = register_instance(self)

    def get_or_build_kernel(
        self,
        name: str,
        inputs: "Sequence[torch.Tensor | None]",
        *,
        key: Hashable = None,
        build: Optional[Callable[[], _Entry]] = None,
    ) -> _Entry:
        """Return the kernel for this call, building it once on a miss.

        The Op layer's only get-or-build, and the one place the two implementations fork.

        Args:
            name: Which of this op's kernels is being asked for.
            inputs: The tensors this kernel will be handed, one slot per
                ``signature.inputs`` entry, in that order. An ``optional: true`` input the
                call did not pass occupies its slot as ``None`` — the same value ``forward``
                was handed, so presence is a fact the builder reads off the slot rather than
                off how many slots there are. An external target needs *inputs*; omitting
                them leaves this op in-tree only.
            key: What the *in-tree* kernel specializes on, typically
                ``(self._cache_key(*input_shapes), dtype)`` or just the dtype. The external
                path keys on the input signature instead.
            build: How the *in-tree* kernel is constructed, called once per key. See
                ``Op._entry_kernels`` for what it may return.

        Returns:
            The stored entry, identical across calls describing the same specialization.

        Raises:
            OpNotAvailableError: A target serves this op but the call site handed over no
                tensor at all; or there is no in-tree implementation and no target.
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

        settled_here = self._builder is _UNRESOLVED
        if settled_here:
            # ``__call__`` settled this already — unless it was traced. Dynamo defers a
            # traced frame's attribute writes until after the graph has run, so a
            # ``forward`` behind the compile boundary arrives here still ``_UNRESOLVED``
            # and would take the in-tree path on the very call that chose a target.
            self._resolve_builder(tuple(inputs), {})

        try:
            builder = self._builder
            if builder is None or builder is _UNRESOLVED:
                # In-tree: the op knows what its own kernel specializes on, so it says.
                if build is None:
                    raise OpNotAvailableError(
                        f"{type(self).__name__} has no in-tree implementation for {name!r}, "
                        f"so it needs a target that registers one; known targets for this "
                        f"op: {registered_targets(type(self).__name__)}"
                    )
                if key not in entries:
                    entries[key] = build()
                return entries[key]

            # External: this layer cannot know what the target's kernel specializes on, so
            # it keys on every cheap fact it has: the dtype and shape of each input.
            specs = tuple(None if t is None else TensorSpec.of(t) for t in inputs)
            present = tuple(spec for spec in specs if spec is not None)
            if not present:
                raise OpNotAvailableError(
                    f"target {self._settled_target!r} serves {type(self).__name__}, but its "
                    f"{name!r} call site does not hand over the tensors a builder is "
                    f"described with; that op is not wired to external targets yet"
                )
            # The device is part of the key: a kernel built for one of a target's devices
            # may hold resources allocated on it. The op layer has already checked that
            # this call's tensors agree on a device, so the first one speaks for all.
            # An absent optional input keeps its place as ``None``: drop it and a clamp
            # with only a lower bound describes itself exactly like one with only an
            # upper bound, so the second is served the first one's kernel.
            signature = (present[0].device,) + tuple(
                None if spec is None else (spec.dtype, spec.shape) for spec in specs
            )
            if signature not in entries:
                entries[signature] = self._build_external(builder, name, specs)
            return entries[signature]
        except Exception:
            # Whoever settled it unsettles it. ``__call__``'s handler does not run when
            # the failure comes out of a compiled graph, so this one has to.
            if settled_here:
                self._unsettle()
            raise

    def _build_external(
        self,
        builder: BuildKernel,
        name: str,
        specs: "tuple[TensorSpec | None, ...]",
    ) -> object:
        """Ask the target for a kernel and hold it to the one rule this boundary has.

        *specs* carries one slot per ``signature.inputs`` entry; an absent optional input's
        slot is ``None``.
        """
        kernel = builder(*specs, **self._manifest_params())
        if not callable(kernel):
            raise OpNotAvailableError(
                f"target {self._settled_target!r} built {kernel!r} for "
                f"{type(self).__name__}.{name}, which is not callable; a builder returns "
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
        """Return a read-only view of the entries built for *role* so far.

        Empty before the role's first build. For introspection — tests,
        benchmark reporting — never for dispatch: an execution path asks
        ``get_or_build_kernel`` so a miss builds rather than raises.
        """
        roles = getattr(self, "_kernel_roles", None) or {}
        return MappingProxyType(roles.get(role, {}))

    def kernel_delegates(self) -> Sequence["Op"]:
        """Return the ops whose kernels this op runs.

        A composite op — one that resolves its call through another op rather
        than building the kernel itself — overrides this so enumeration reaches
        the delegate. Default: this op builds everything it runs.
        """
        return ()

    def run_config(self) -> Optional[dict]:
        """The configuration the op's kernels were built with, or ``None``.

        An op reports what it ran; a caller reaches it by calling the op, not by
        reading the attributes of the kernels behind it. An op given a config of
        its own answers with it; otherwise the kernels it built do, and those
        share their dtype and op kind for one call, so the first configured one
        answers for the call.
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
        """Yield every kernel the op holds, each one once.

        Enumeration is explicit: the entries of every role, then ``self.kernel``
        for an op that binds one directly, then the same walk over each
        ``kernel_delegates()`` entry. A kernel bound to any other attribute is
        not searched for — an op that holds one builds it through a role.
        """
        seen: set[int] = set()
        for kernel in self._walk_kernels():
            if id(kernel) not in seen:
                seen.add(id(kernel))
                yield kernel

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

    def _walk_kernels(self) -> Iterator[Kernel]:
        """Yield the kernels this op and its delegates hold, duplicates included."""
        for op in self._walk_ops():
            for entries in (getattr(op, "_kernel_roles", None) or {}).values():
                for entry in entries.values():
                    yield from self._entry_kernels(entry)
            yield from self._entry_kernels(getattr(op, "kernel", None))

    def autotune(self) -> None:
        """Put the op in tuned mode: what it holds now, and what it builds next.

        Tuning is a lifecycle decision, not a property of one kernel, so it
        applies to specializations that do not exist yet — an op tuned before
        its first fp16 call is tuned when bf16 arrives later. Setting ``tune``
        is what carries it: a factory reads the flag when it runs, so the
        kernel it builds tunes itself, the same way ``tune=True`` at
        construction does.
        """
        for op in self._walk_ops():
            op.tune = True
        for kernel in self.iter_kernels():
            kernel.autotune()

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Run the op."""
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Make the op callable.

        Settles which set of kernels serves this instance, once, then delegates to
        ``forward`` — which is the same for every target. The only fork is inside
        `get_or_build_kernel`, which settles it a second time when this settling
        could not reach it; see there.

        A call that fails settles nothing. Otherwise one invalid call would aim the instance
        for good: ``op(x_cpu, weight_cuda)`` picks a target from the first tensor, then
        ``forward`` rejects the mismatch, and every later call would go where that one
        pointed.
        """
        if self._builder is not _UNRESOLVED:
            return self.forward(*args, **kwargs)

        self._resolve_builder(args, kwargs)
        try:
            return self.forward(*args, **kwargs)
        except Exception:
            self._unsettle()
            raise

    def _refuse_empty_input(self, inputs: "Sequence[torch.Tensor | None]") -> None:
        """Raise for a call whose every declared output would hold no elements.

        Such a call leaves the launch a zero-sized grid, which reports itself as an
        internal assertion saying nothing about what is unsupported.

        Kernel selection hosts this because both the eager and the traced path reach it.

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
        """Undo a settling whose call did not finish, dropping what it built."""
        self._builder = _UNRESOLVED
        self._settled_target = None
        self._kernel_roles = {}

    def _resolve_builder(self, args: tuple, kwargs: dict) -> None:
        """Decide which target serves this instance and remember its builder.

        Once decided it does not change: the kernels this instance has built belong to that
        target. An instance is therefore bound to that target's devices — handing it tensors
        from elsewhere is a caller error, and the kernel is what reports it. A call carrying
        no tensor probes no device and decides nothing.

        Raises:
            OpNotAvailableError: The selected target registers no builder for this op.
        """
        device = self._first_tensor_device(args, kwargs)
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

        Default implementation returns the tuple of non-static-axis sizes across
        all input shapes, using ``self._static_axes`` to decide which axes are
        committed at ctor. This is always correct for any Op, but may
        over-fragment the kernel cache when ``_static_axes`` is empty (one
        compile per distinct input shape).

        Override in subclasses to project the shape onto whatever the kernel
        actually depends on — for example, flattening leading dims to a single
        product when the kernel treats input as 2D.

        When ``_static_axes`` is empty AND the subclass does not override
        ``_cache_key``, a ``UserWarning`` is emitted once per subclass type to
        surface the missing override.
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
