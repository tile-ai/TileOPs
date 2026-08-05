import warnings
from abc import ABC, abstractmethod
from typing import Hashable, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

from .compile_boundary import register_instance

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()

def _iter_kernels(value: object, _depth: int = 0) -> "list[Kernel]":
    """Return the kernels *value* holds, descending through dicts and sequences.

    A kernel cache is a dict keyed by shape and dtype; an op that builds
    several kernels together (preprocess plus backward, say) stores a tuple
    per entry, so the search has to go two levels down.
    """
    if isinstance(value, Kernel):
        return [value]
    if _depth >= 2:
        return []
    if isinstance(value, dict):
        return [k for v in value.values() for k in _iter_kernels(v, _depth + 1)]
    if isinstance(value, (tuple, list)):
        return [k for v in value for k in _iter_kernels(v, _depth + 1)]
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
        >>> output, lse = op(Q, K, V)
        >>> op.check()  # Verify correctness
        >>> latency = op.profile()  # Benchmark performance

    Attributes:
        kernel: single kernel, for ops that hold one; ops that build per
            dtype keep a cache instead
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
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None

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
    # Why: marking them abstract today breaks every op under tileops/ops/ that has
    #     not been migrated to docs/design/ops-design.md, and eval_roofline bodies
    #     are emitted by scaffold-op codegen that has not run for most ops yet. The
    #     trust model requires one migration PR per op.
    # Cleanup: when all three are implemented across tileops/ops/, make all three
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
        """Validate and install the resolved kernel map onto ``self.kernel_map``.

        Iterates ``self.default_kernel_map`` and, for each entry, picks the
        override from ``candidate_map`` when present, falling back to the
        default. Each resolved kernel is then validated against the current
        device architecture: if the kernel declares ``supported_archs`` and the
        current ``sm`` version is not in that list, a ``ValueError`` is raised
        — the same exception class produced by the auto-discovery path on the
        same input. Both auto-discovered and user-supplied maps share this
        single validate-and-install path, so arch-compat checks fire
        identically regardless of provenance.
        """
        default_map = self.default_kernel_map
        if default_map is None or len(default_map) == 0:
            # Composite op: store override verbatim; sub-ops enforce arch-compat themselves.
            self.kernel_map = dict(candidate_map) if candidate_map else {}
            return
        resolved: dict[str, Kernel] = {}
        current_arch = get_sm_version()
        for name, default_kernel in default_map.items():
            if candidate_map is not None and name in candidate_map:
                kernel_type = candidate_map[name]
            else:
                kernel_type = default_kernel
            if (
                kernel_type is not None
                and kernel_type.supported_archs is not None
                and current_arch not in kernel_type.supported_archs
            ):
                raise ValueError(
                    f'{kernel_type.__name__} is not supported on architecture {current_arch}')
            resolved[name] = kernel_type
        self.kernel_map = resolved

    def dispatch_kernel(self, kernel_map: Optional[dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map (auto-discovery entry point)."""
        self._install_kernel_map(kernel_map)
        # Conforming __init__s all pass through here — the zero-boilerplate
        # registration point for the compile dispatch boundary.
        self._instance_key = register_instance(self)

    def autotune(self) -> None:
        """Autotune every kernel the op holds.

        An op that builds its kernels at forward time keeps them in a cache
        keyed by shape and dtype, so the search covers the values of dict
        attributes and the items of tuple/list attributes as well as kernels
        bound directly. Only kernels already built are tuned; a cache the op
        has not populated yet has nothing to tune.
        """
        seen: set[int] = set()
        for attr_name in dir(self):
            # ``__dict__`` would re-yield every kernel bound as an attribute.
            if attr_name.startswith("__"):
                continue
            for kernel in _iter_kernels(getattr(self, attr_name, None)):
                if id(kernel) not in seen:
                    seen.add(id(kernel))
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
