import dataclasses
import warnings
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Callable, Hashable, Iterator, Mapping, Optional, Sequence, TypeVar, Union

import torch

from tileops.kernels.kernel_base import Kernel

from .compile_boundary import register_instance

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()

_Entry = TypeVar("_Entry")


def _entry_kernels(entry: object) -> "list[Kernel]":
    """Return the kernels one entry holds.

    An entry is a kernel, a sequence of kernels built together, or a dataclass
    carrying them alongside what else the specialization implies. An entry that
    hides its kernels from this walk is invisible to ``autotune``.
    """
    if isinstance(entry, Kernel):
        return [entry]
    if isinstance(entry, (tuple, list)):
        return [k for item in entry for k in _entry_kernels(item)]
    if dataclasses.is_dataclass(entry) and not isinstance(entry, type):
        return [k for f in dataclasses.fields(entry)
                for k in _entry_kernels(getattr(entry, f.name))]
    return []

class Op(ABC):
    """Base class for TileOPs operations.

    A Op represents a computational operation with:
    - Hardware-aware kernel dispatch
    - Correctness testing via reference implementation
    - Performance profiling
    - Autotuning interface

    Examples:
        >>> from tileops.ops import MultiHeadAttentionFwdOp
        >>> op = MultiHeadAttentionFwdOp(batch=1, heads=8, seq_len=512, dim=64, is_causal=True)
        >>> Q, K, V = op.gen_inputs()
        >>> output = op(Q, K, V)
        >>> op.check()  # Verify correctness
        >>> latency = op.profile()  # Benchmark performance

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

    kernel: Kernel
    kernel_map: Optional[dict[str, Kernel]] = None
    # Built entries, ``{role: {key: entry}}``. Annotation only: the instance
    # attribute appears on the first ``get_or_build_kernel`` call, so an op that
    # has built nothing carries no dict, and no constructor declares one.
    _kernel_roles: dict[str, dict[Hashable, object]]
    # Dispatch keys the caller replaced through ``kernel_map=``.
    _overridden_keys: frozenset = frozenset()
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
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
        from the subclass's manifest entry. Each codegen pass is a no-op
        when the subclass does not advertise manifest metadata, supplies
        its own override, or is marked ``status: spec-only``. Codegen
        modules are lazy-imported to avoid a circular import at ``Op``
        definition time.
        """
        super().__init_subclass__(**kwargs)
        from tileops.ops._dtype_codegen import maybe_install_validator
        from tileops.ops._roofline_codegen import maybe_install_eval_roofline
        maybe_install_validator(cls)
        maybe_install_eval_roofline(cls)

    # FIXME(staged-rollout): the three contract stubs below — _infer_output_shapes,
    # _validate_dtypes and eval_roofline — raise NotImplementedError instead of
    # being @abstractmethod.
    #
    # Broken invariant: L1 does not enforce that every concrete Op implements them.
    # Why: marking them abstract today breaks every op under src/tileops/ops/ that has
    #     not been migrated to docs/design/ops-design.md, and eval_roofline bodies
    #     are emitted by scaffold-op codegen that has not run for most ops yet. The
    #     trust model requires one migration PR per op.
    # Cleanup: when all three are implemented across src/tileops/ops/, make all three
    #     @abstractmethod and delete this marker.

    @property
    @abstractmethod
    def default_kernel_map(self) -> dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def _infer_output_shapes(self, **shape_kwargs: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        """Infer output tensor shapes from input shapes.

        Concrete ops override this with a signature matching the named input
        shapes declared in their manifest ``shape_rules`` section (e.g.
        ``_infer_output_shapes(self, x_shape, weight_shape)``). The uniform
        ``**shape_kwargs`` base signature exists only to make the L1 contract
        grepable and discoverable; see docs/design/ops-design.md §``_infer_output_shapes``.
        """
        raise NotImplementedError(
            "_infer_output_shapes must be implemented by the concrete Op subclass; "
            "see docs/design/ops-design.md §`_infer_output_shapes` (codegen)")

    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        """Validate dtypes of input tensors passed to ``forward``.

        Concrete ops override this with a signature matching their manifest
        ``signature.inputs`` (e.g. ``_validate_dtypes(self, x, weight)``).
        See docs/design/ops-design.md §``_validate_dtypes``.
        """
        raise NotImplementedError(
            "_validate_dtypes must be implemented by the concrete Op subclass; "
            "see docs/design/ops-design.md §`_validate_dtypes` (codegen)")

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
            "docs/design/roofline.md §4.4.6 (Evaluator Surface Boundary)")

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

        A composite op must not pass its own resolved ``kernel_map`` down: every
        key would arrive at the sub-op looking replaced, and a replacement that
        cannot serve a call is refused rather than passed over — so a default
        the sub-op would have selected around becomes an error. Only what the
        caller actually supplied is an override.
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
                    "the kernel supplied for " + "; ".join(refused_overrides)
                    + f" — selection does not fall back to the shipped '{chosen[0]}' "
                    f"when a replacement is in force. Call: {call}")
            return chosen[0]
        if not chosen:
            lead = ("the kernel supplied for " + "; ".join(refused_overrides) + ", and "
                    if refused_overrides else "")
            raise ValueError(
                lead + "no implementation serves this call: "
                + "; ".join(rejected or ["no implementation is installed"])
                + f". Call: {call}")
        raise ValueError(
            f"dispatch is ambiguous: {', '.join(chosen)} all serve this call, so none "
            f"is the answer. Implementations of one key must serve disjoint regions, "
            f"and at most one of them may be general. Call: {call}")

    def dispatch_kernel(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map (auto-discovery entry point)."""
        self._install_kernel_map(kernel_map)
        # Conforming __init__s all pass through here — the zero-boilerplate
        # registration point for the compile dispatch boundary.
        self._instance_key = register_instance(self)

    def get_or_build_kernel(
        self,
        role: str,
        key: Hashable,
        factory: Callable[[], _Entry],
    ) -> _Entry:
        """Return the entry at ``(role, key)``, calling *factory* once on a miss.

        This is the Op layer's only get-or-build: an op names the role it is
        asking about and the specialization it wants, and supplies the one thing
        that is its own — how the kernel is constructed. Slots are created on
        first use, so no constructor declares one.

        Args:
            role: The kernel role this entry plays. Conventionally
                the ``kernel_map`` dispatch key whose kernel *factory* builds.
            key: What the entry is specialized for, typically
                ``(self._cache_key(*input_shapes), dtype)`` or just the dtype.
            factory: Zero-argument callable building the entry, called only on
                a miss. See ``_entry_kernels`` for what an entry may hold.

        Returns:
            The stored entry, identical across calls with the same
            ``(role, key)``.
        """
        # Plain attribute reads and dict lookups, no ``self.__dict__``: this
        # runs inside a dynamo-traced forward on every cache hit, and dynamo
        # cannot trace a method call on an instance ``__dict__``.
        roles = getattr(self, "_kernel_roles", None)
        if roles is None:
            roles = {}
            self._kernel_roles = roles
        entries = roles.get(role)
        if entries is None:
            entries = {}
            roles[role] = entries
        if key not in entries:
            entries[key] = factory()
        return entries[key]

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
                    yield from _entry_kernels(entry)
            yield from _entry_kernels(getattr(op, "kernel", None))

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
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, tuple]:
        """Make the op callable - delegates to forward()"""
        return self.forward(*args, **kwargs)

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
