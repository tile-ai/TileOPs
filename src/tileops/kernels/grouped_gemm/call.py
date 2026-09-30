"""The facts of one grouped-GEMM call, and the kernel interface its implementations inherit."""

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, KernelInterface

__all__ = ["GroupedGemmCall", "GroupedGemmFwdInterface", "grouped_gemm_entry"]


@dataclasses.dataclass(frozen=True)
class GroupedGemmCall(CallSpec):
    """One grouped GEMM over a tight, per-group row layout.

    ``numel`` and ``num_experts`` are the declared spread, not the routed one: the
    routing lands on the device, so a region over these holds for every call.
    """

    numel: int = 0
    num_experts: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    transpose_a: bool = False
    transpose_b: bool = True


class GroupedGemmFwdInterface(KernelInterface):
    """One GEMM per group, with the groups packed along a single axis.

    ``transpose_a`` says which axis the groups split: rows of *a* (NT / NN), or the
    contraction (TN / TT). ``transpose_b`` says whether the per-group operand is stored
    transposed.
    """

    request = GroupedGemmCall

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Run every group's GEMM; nothing is written in place.

        Every tensor is contiguous on ``call.device``, the operands in ``call.dtype``.
        The metadata values are the caller's obligation and are not checked, since
        checking them would synchronise: ``batch_sizes`` sums to ``call.numel`` and
        ``batch_offsets`` is its exclusive prefix sum.

        Args:
            a: ``(call.numel, call.k)``, or ``(call.numel, call.n)`` when
                ``call.transpose_a``.
            b: ``(call.num_experts, call.n, call.k)`` when ``call.transpose_b``, else
                ``(call.num_experts, call.k, call.n)``; with ``call.transpose_a``,
                ``(call.k, call.numel)`` or ``(call.numel, call.k)``.
            batch_sizes: ``(call.num_experts,)`` ``int32`` rows per group.
            batch_offsets: ``(call.num_experts,)`` ``int32`` start row of each group.

        Returns:
            A new tensor in ``call.dtype``: ``(call.numel, call.n)``, or
            ``(call.num_experts, call.n, call.k)`` when ``call.transpose_a``.
        """


def grouped_gemm_entry(cls: type, call: GroupedGemmCall) -> Entry:
    """The entry for a grouped-GEMM candidate: both take the same construction arguments.

    The device index is in the identity because the kernel is compiled for the
    architecture it is built on.
    """
    index = call.device.index if call.device is not None else None
    identity = (
        call.numel,
        call.num_experts,
        call.n,
        call.k,
        call.dtype,
        call.transpose_a,
        call.transpose_b,
        index,
    )
    return identity, lambda: cls(
        call.numel,
        call.num_experts,
        call.n,
        call.k,
        call.dtype,
        transpose_a=call.transpose_a,
        transpose_b=call.transpose_b,
        device_index=index,
    )
