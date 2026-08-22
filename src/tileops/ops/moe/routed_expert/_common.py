"""What the routed-expert ops taking a grouped operand pair share.

``MoeGateUpFwdOp`` and ``MoeGroupedGemmNopadFwdOp`` take the same four inputs --
the two operands plus the group sizes and offsets -- so the eager path is the
same for both: check the dtypes, check the operands agree on a device, make them
contiguous, resolve the kernel and launch it.
"""

import torch

__all__ = ["GroupedOperandEagerForward"]


class GroupedOperandEagerForward:
    """The eager path for an op called as ``(a, b, true_sizes, true_offsets)``.

    A subclass supplies ``_validate_dtypes`` (the manifest codegen installs it)
    and ``_get_kernel``.
    """

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        true_sizes: torch.Tensor,
        true_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Validate, normalize, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo
        cannot follow.
        """
        self._validate_dtypes(a, b, true_sizes, true_offsets)
        for name, t in (("b", b), ("true_sizes", true_sizes), ("true_offsets", true_offsets)):
            if t.device != a.device:
                raise ValueError(f"{name} must be on {a.device}, got {t.device}")
        self.dtype = a.dtype
        inputs = tuple(t.contiguous() for t in (a, b, true_sizes, true_offsets))
        return self._get_kernel(inputs, a.dtype)(*inputs)
