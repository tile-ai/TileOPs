import ast
import warnings
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, Hashable, List, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()

# ``ast.Num`` is a deprecated alias (Python 3.8+) that becomes ``ast.Constant``
# under the hood; AC-4 requires the roofline evaluator to accept legacy
# ``Num`` nodes for back-compat. Resolve it here without triggering the
# Python 3.12+ DeprecationWarning, and omit it from the whitelist tuple when
# ``ast.Constant`` already covers it (which is true on every supported Python).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    _AST_NUM: type = getattr(ast, "Num", ast.Constant)

# Whitelist of AST nodes permitted in roofline expressions.
# Explicitly excludes Call, Attribute, Subscript, Lambda, Comprehension, etc.
# ``ast.Num`` nodes (legacy) are instances of ``ast.Constant`` on Python 3.8+,
# so they pass the Constant gate without needing a separate whitelist entry.
_SAFE_AST_NODES: Tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    _AST_NUM,  # ast.Num (back-compat); subclass of Constant on Python 3.8+
    ast.Name,
    ast.Load,
    # Arithmetic operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Unary operators
    ast.UAdd,
    ast.USub,
)


def _safe_eval(expr: str, ctx: Dict[str, Union[int, float]]) -> Union[int, float]:
    """Evaluate an arithmetic expression against ``ctx`` using a whitelist AST walker.

    Only binary/unary arithmetic on numeric constants and names resolved from
    ``ctx`` is permitted. Function calls, attribute access, subscripts, lambdas,
    comprehensions, and any other node type raise :class:`ValueError` naming
    the forbidden node.

    This is the roofline evaluator referenced by :meth:`Op.eval_roofline`; it
    intentionally avoids :func:`eval` / :func:`exec` and any third-party
    expression evaluator.
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"invalid roofline expression {expr!r}: {e}") from e

    def _walk(node: ast.AST) -> Union[int, float]:
        # Suppress the Python 3.12+ DeprecationWarning triggered by ast.Num
        # appearing inside the isinstance tuple. The ast.Num entry is a legacy
        # alias of ast.Constant on all supported Pythons; keeping it in the
        # whitelist matches AC-4 and the design doc without changing behavior.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            permitted = isinstance(node, _SAFE_AST_NODES)
        if not permitted:
            raise ValueError(
                f"forbidden AST node {type(node).__name__} in roofline expression "
                f"{expr!r}; only arithmetic over Name/Constant is permitted")
        if isinstance(node, ast.Expression):
            return _walk(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError(
                    f"forbidden constant type {type(node.value).__name__} in "
                    f"roofline expression {expr!r}")
            return node.value
        # Note: on Python 3.8+, ``ast.Num`` nodes are instances of
        # ``ast.Constant`` and are handled by the Constant branch above.
        if isinstance(node, ast.Name):
            if node.id not in ctx:
                raise ValueError(
                    f"undefined name {node.id!r} in roofline expression {expr!r}")
            return ctx[node.id]
        if isinstance(node, ast.UnaryOp):
            operand = _walk(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(
                f"forbidden unary operator {type(node.op).__name__} in "
                f"roofline expression {expr!r}")
        if isinstance(node, ast.BinOp):
            left = _walk(node.left)
            right = _walk(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return left + right
            if isinstance(op, ast.Sub):
                return left - right
            if isinstance(op, ast.Mult):
                return left * right
            if isinstance(op, ast.Div):
                return left / right
            if isinstance(op, ast.FloorDiv):
                return left // right
            if isinstance(op, ast.Mod):
                return left % right
            if isinstance(op, ast.Pow):
                return left**right
            raise ValueError(
                f"forbidden binary operator {type(op).__name__} in "
                f"roofline expression {expr!r}")
        # Unreachable: _SAFE_AST_NODES gate above covers every handled type.
        raise ValueError(  # pragma: no cover - defensive
            f"unhandled AST node {type(node).__name__} in roofline expression {expr!r}")

    return _walk(tree)


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
    kernel_map: Optional[Dict[str, Kernel]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None

    # Set of (input_index, axis) pairs identifying static (ctor-committed) axes.
    # `input_index` is the position in *input_shapes; `axis` is a non-negative
    # axis index within that shape. Subclasses set this to reflect their
    # manifest `static_dims`. Default empty = no committed axes.
    _static_axes: FrozenSet[Tuple[int, int]] = frozenset()

    # Roofline evaluation slots (see docs/ops-design.md §`eval_roofline`).
    # Concrete ops override these as class-level declarations; `eval_roofline`
    # reads them and evaluates the expressions via `_safe_eval`. Defaults
    # produce `(0, 0)` — a transitional escape hatch for ops not yet migrated.
    _roofline_vars: List[str] = []
    _flops_expr: str = "0"
    _bytes_expr: str = "0"

    @property
    @abstractmethod
    def default_kernel_map(self) -> Dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def _infer_output_shapes(self, **shape_kwargs: Tuple[int, ...]) -> Dict[str, tuple]:
        """Infer output tensor shapes from input shapes.

        Concrete ops override this with a signature matching the named input
        shapes declared in their manifest ``shape_rules`` section (e.g.
        ``_infer_output_shapes(self, x_shape, weight_shape)``). The uniform
        ``**shape_kwargs`` base signature exists only to make the L1 contract
        grepable and discoverable; see docs/ops-design.md §``_infer_output_shapes``.
        """
        # FIXME(staged-rollout): L1 Op does not yet strictly enforce _infer_output_shapes
        # via @abstractmethod; base raises NotImplementedError instead.
        #
        # Broken invariant: L1 base does not strictly enforce implementation
        #     of _infer_output_shapes on every concrete Op subclass.
        # Why: Introducing @abstractmethod now would break all existing concrete
        #     ops under tileops/ops/ that have not yet been migrated to the spec
        #     in docs/ops-design.md; the trust model requires a separate
        #     per-op migration PR.
        # Cleanup: once all concrete ops under tileops/ops/ implement both
        #     _infer_output_shapes and _validate_dtypes, convert this stub
        #     (and _validate_dtypes below) to `@abstractmethod` and remove the
        #     default class-level _roofline_vars/_flops_expr/_bytes_expr
        #     transitional defaults.
        raise NotImplementedError(
            "_infer_output_shapes must be implemented by the concrete Op subclass; "
            "see docs/ops-design.md §`_infer_output_shapes` (codegen)")

    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        """Validate dtypes of input tensors passed to ``forward``.

        Concrete ops override this with a signature matching their manifest
        ``signature.inputs`` (e.g. ``_validate_dtypes(self, x, weight)``).
        See docs/ops-design.md §``_validate_dtypes``.
        """
        # FIXME(staged-rollout): L1 Op does not yet strictly enforce _validate_dtypes
        # via @abstractmethod; base raises NotImplementedError instead.
        #
        # Broken invariant: L1 base does not strictly enforce implementation
        #     of _validate_dtypes on every concrete Op subclass.
        # Why: Introducing @abstractmethod now would break all existing concrete
        #     ops under tileops/ops/ that have not yet been migrated to the spec
        #     in docs/ops-design.md; the trust model requires a separate
        #     per-op migration PR.
        # Cleanup: once all concrete ops under tileops/ops/ implement both
        #     _infer_output_shapes and _validate_dtypes, convert this stub
        #     (and _infer_output_shapes above) to `@abstractmethod`.
        raise NotImplementedError(
            "_validate_dtypes must be implemented by the concrete Op subclass; "
            "see docs/ops-design.md §`_validate_dtypes` (codegen)")

    def eval_roofline(self) -> Tuple[int, int]:
        """Evaluate (flops, bytes) from class-level roofline slots.

        Reads ``_roofline_vars`` (list of attribute names on ``self``) plus a
        derived ``elem_bytes`` from ``self.dtype``, then evaluates
        ``_flops_expr`` and ``_bytes_expr`` via the whitelist AST evaluator
        :func:`_safe_eval`. Returns integer ``(flops, bytes)``.

        When the class-level slots are at defaults, returns ``(0, 0)`` — a
        transitional default for ops not yet migrated. See
        docs/ops-design.md §``eval_roofline``.
        """
        ctx: Dict[str, Union[int, float]] = {}
        for name in self._roofline_vars:
            ctx[name] = getattr(self, name)
        if self.dtype is not None:
            ctx["elem_bytes"] = torch.tensor([], dtype=self.dtype).element_size()
        else:
            ctx["elem_bytes"] = 0
        flops = _safe_eval(self._flops_expr, ctx)
        nbytes = _safe_eval(self._bytes_expr, ctx)
        return int(flops), int(nbytes)

    def dispatch_kernel(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        if self.default_kernel_map is None or len(self.default_kernel_map) == 0:
            raise ValueError("default_kernel_map must be non-empty")
        self.kernel_map = {}
        for name, default_kernel in self.default_kernel_map.items():
            if kernel_map is not None and name in kernel_map:
                kernel_type = kernel_map[name]
            else:
                kernel_type = default_kernel
            current_arch = get_sm_version()
            if kernel_type is not None and current_arch not in kernel_type.supported_archs:
                raise ValueError(
                    f'{kernel_type.__name__} is not supported on architecture {current_arch}')
            self.kernel_map[name] = kernel_type

    def autotune(self) -> None:
        """Autotune all kernels of the op"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        """Make the op callable - delegates to forward()"""
        return self.forward(*args, **kwargs)

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
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
