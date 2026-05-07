import warnings
from abc import ABC, abstractmethod
from typing import Hashable, Optional, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()

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
        kernel: top.Kernel instance (e.g. mha_fwd_kernel)
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
        # FIXME(staged-rollout): L1 Op does not yet strictly enforce _infer_output_shapes
        # via @abstractmethod; base raises NotImplementedError instead.
        #
        # Broken invariant: L1 base does not strictly enforce implementation
        #     of _infer_output_shapes on every concrete Op subclass.
        # Why: Introducing @abstractmethod now would break all existing concrete
        #     ops under tileops/ops/ that have not yet been migrated to the spec
        #     in docs/design/ops-design.md; the trust model requires a separate
        #     per-op migration PR.
        # Cleanup: once all concrete ops under tileops/ops/ implement
        #     _infer_output_shapes, _validate_dtypes, and eval_roofline,
        #     convert this stub (and the two below) to `@abstractmethod`.
        raise NotImplementedError(
            "_infer_output_shapes must be implemented by the concrete Op subclass; "
            "see docs/design/ops-design.md §`_infer_output_shapes` (codegen)")

    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        """Validate dtypes of input tensors passed to ``forward``.

        Concrete ops override this with a signature matching their manifest
        ``signature.inputs`` (e.g. ``_validate_dtypes(self, x, weight)``).
        See docs/design/ops-design.md §``_validate_dtypes``.
        """
        # FIXME(staged-rollout): L1 Op does not yet strictly enforce _validate_dtypes
        # via @abstractmethod; base raises NotImplementedError instead.
        #
        # Broken invariant: L1 base does not strictly enforce implementation
        #     of _validate_dtypes on every concrete Op subclass.
        # Why: Introducing @abstractmethod now would break all existing concrete
        #     ops under tileops/ops/ that have not yet been migrated to the spec
        #     in docs/design/ops-design.md; the trust model requires a separate
        #     per-op migration PR.
        # Cleanup: once all concrete ops under tileops/ops/ implement
        #     _infer_output_shapes, _validate_dtypes, and eval_roofline,
        #     convert this stub (and the others) to `@abstractmethod`.
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
        # FIXME(staged-rollout): L1 Op does not yet strictly enforce eval_roofline
        # via @abstractmethod; base raises NotImplementedError instead.
        #
        # Broken invariant: L1 base does not strictly enforce implementation
        #     of eval_roofline on every concrete Op subclass.
        # Why: Introducing @abstractmethod now would break every existing
        #     concrete op under tileops/ops/ (none of them ship an
        #     eval_roofline yet). The scaffold-op codegen work that will
        #     generate these bodies per docs/design/roofline.md §4.4 is pre-
        #     requisite; the trust model requires a separate per-op migration
        #     PR to flip any given op from stub to generated body.
        # Cleanup: once all concrete ops under tileops/ops/ implement
        #     eval_roofline (via codegen emission per docs/design/roofline.md §4.4),
        #     convert this stub and the two stubs above (_infer_output_shapes,
        #     _validate_dtypes) to `@abstractmethod`.
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
            raise ValueError("default_kernel_map must be non-empty")
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

    def autotune(self) -> None:
        """Autotune all kernels of the op"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

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
