"""What crosses the boundary between TileOPs and a backend. Types only."""

from __future__ import annotations

from typing import Callable, Final, NamedTuple, Optional, Union

import torch


class TensorSpec(NamedTuple):
    """What one tensor is, without the tensor. Handed to ``build_kernel``."""

    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]

    @staticmethod
    def of(tensor: torch.Tensor) -> "TensorSpec":
        """Describe *tensor*."""
        return TensorSpec(tensor.device, tensor.dtype, tuple(tensor.shape))


# One call's result: the op's declared outputs, a declared nullable one as ``None``. An op
# whose outputs are all written into declared mutated inputs returns ``None``:
# ``torch.library.custom_op`` cannot express a return value aliasing an input.
KernelResult = Union[torch.Tensor, tuple[Optional[torch.Tensor], ...], None]

# Called ``build_kernel(*inputs, **params)``: a `TensorSpec` per slot of the op's forward
# inputs — the manifest's ``signature.inputs`` in order, then its ``resources.workspaces``
# — ``None`` for an ``optional: true`` input the call did not pass, so presence is read
# off the slot rather than off how many slots there are; then ``signature.params`` by
# keyword. Both lists are per-op, which the type system cannot express, hence ``...``.
#
# The callable it returns serves the whole op and is called with exactly those tensors.
# It returns the declared outputs, or writes the declared mutated outputs and returns
# ``None``; a workspace is scratch it may use or ignore. It checks the preconditions that
# need the tensors' values, such as an index range, and raises ``ValueError`` itself; the
# op checks shapes, dtypes and params before asking. For an op with autograd, the tensors
# it returns carry the op's backward.
BuildKernel = Callable[..., Callable[..., KernelResult]]

# "Is this the kind of device my kernels are written for" — ``False``, not an exception,
# for the rest. Per-call support is ``build_kernel``'s answer; it sees the dtypes too.
DetectFn = Callable[[torch.device], bool]


class _Builtin:
    """The type of :data:`BUILTIN`. One instance, compared by identity."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "BUILTIN"


# Ask for the in-tree implementation whatever is installed. Not a target name: unregistered
# and never in ``registered_targets()``.
BUILTIN: Final = _Builtin()

# What ``target=`` and the process default accept.
Target = Union[str, _Builtin, None]
