"""The reduction family's compile boundary: one opaque operator per op.

Every op in this family takes one tensor, so what differs between them is only how many
tensors come back. Two factories cover that, and both put the boundary in the op layer:
the node in a traced graph is the op's, which is what keeps the graph the same when
another target serves the op.

The fake reads its shape from the op's ``_infer_output_shapes`` and its dtype from the
manifest, never from a kernel class. The operator's name comes from the manifest too — the
family and the entry key — so ``compile_op_names`` and the name cannot drift apart.
"""

import re

import torch

from tileops.manifest import load_manifest

from .._output_dtype import resolve_output_dtype
from ..compile_boundary import get_instance

__all__ = ["register_reduction_op"]


def _snake(name: str) -> str:
    """``"VarMeanFwdOp"`` -> ``"var_mean_fwd"``."""
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", spaced).lower().removesuffix("_op")


def register_reduction_op(op_cls) -> None:
    """Register *op_cls* as one opaque operator and name it on the class.

    Sets ``op_cls._wrapped``, which the op's ``forward`` calls, and
    ``op_cls.compile_op_names``, which lets a test assert the traced graph holds nothing
    else.

    Raises:
        KeyError: *op_cls* has no manifest entry.
        ValueError: The entry declares a number of outputs no factory here covers.
    """
    entry = load_manifest().get(op_cls.__name__)
    if entry is None:
        raise KeyError(
            f"{op_cls.__name__} registers a compile boundary but has no manifest entry; "
            "the operator name, the output names and their dtypes all come from it"
        )
    op_name = f"tileops::{entry['family']}_{_snake(op_cls.__name__)}"
    outputs = tuple(entry["signature"]["outputs"])
    if len(outputs) == 1:
        _register_single(op_cls, op_name, outputs[0])
    elif len(outputs) == 2:
        _register_pair(op_cls, op_name, outputs)
    else:
        raise ValueError(
            f"{op_cls.__name__} declares {len(outputs)} outputs; this family registers one or two"
        )
    op_cls.compile_op_names = (op_name,)


def _register_single(op_cls, op_name: str, output: str) -> None:
    """Register the operator for an op returning one tensor."""

    @torch.library.custom_op(op_name, mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        return get_instance(instance_key)._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: str) -> torch.Tensor:
        op = get_instance(instance_key)
        shapes = op._infer_output_shapes(tuple(x.shape))
        # ``new_empty``, not ``empty_like``: a non-contiguous input's strides must not reach the fake.
        return x.new_empty(
            shapes[output],
            dtype=resolve_output_dtype(op_cls.__name__, x.dtype, output),
        )

    op_cls._wrapped = _wrapped


def _register_pair(op_cls, op_name: str, outputs: "tuple[str, str]") -> None:
    """Register the operator for an op returning two tensors."""
    first, second = outputs

    @torch.library.custom_op(op_name, mutates_args=())
    def _wrapped(x: torch.Tensor, instance_key: str) -> "tuple[torch.Tensor, torch.Tensor]":
        return get_instance(instance_key)._eager_forward(x)

    @_wrapped.register_fake
    def _(x: torch.Tensor, instance_key: str) -> "tuple[torch.Tensor, torch.Tensor]":
        op = get_instance(instance_key)
        shapes = op._infer_output_shapes(tuple(x.shape))
        return (
            x.new_empty(shapes[first], dtype=resolve_output_dtype(op_cls.__name__, x.dtype, first)),
            x.new_empty(
                shapes[second], dtype=resolve_output_dtype(op_cls.__name__, x.dtype, second)
            ),
        )

    op_cls._wrapped = _wrapped
