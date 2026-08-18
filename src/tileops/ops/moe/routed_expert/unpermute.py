"""MoE unpermute op (cutlass path): scatter-add padded expert outputs back to token order."""

from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoeUnpermuteKernel

from ...compile_boundary import get_instance
from ...op_base import Op

__all__ = ["MoeUnpermuteFwdOp"]


class MoeUnpermuteFwdOp(Op):
    """Scatter padded expert outputs back to original token order with weighted reduction.

    Args:
        total_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        padded_batch_sum: Size of the padded mm2_pad buffer (first dim of mm2_pad).
            Must be >= T*K. When used with MoePermuteOp, pass the padded_batch_sum
            value returned by the kernel (T*K + E*block_m upper bound).
            Defaults to total_tokens * top_k for standalone testing only — do NOT
            use the default when mm2_pad comes from MoePermuteOp, as the padded
            buffer will be larger and the kernel will index out of bounds.
        kernel_map: Optional kernel override dict.
        routed_scaling_factor: Scalar applied to the reduced output, folded into
            the unpermute kernel. Defaults to 1.0 (no scaling).

    Example:
        >>> op = MoeUnpermuteFwdOp(total_tokens=4, top_k=2, hidden_size=128, padded_batch_sum=512)
        >>> output = op(mm2_pad, fwd_idx, topk_weights)
    """

    #: Two operators, because ``mutates_args`` is fixed at registration while ``out``
    #: decides per call whether this op writes a caller buffer. Same split as aten's
    #: ``relu`` / ``relu_``: ``forward`` picks one, and a test asserts the graph holds
    #: nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = (
        "top::moe_unpermute_fwd",
        "top::moe_unpermute_fwd_inplace",
    )

    def __init__(
        self,
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        padded_batch_sum: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        routed_scaling_factor: float = 1.0,
    ) -> None:
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.padded_batch_sum = padded_batch_sum if padded_batch_sum is not None else total_tokens * top_k

        self._routed_scaling_factor = routed_scaling_factor
        self.dispatch_kernel(kernel_map)

    def _get_kernel(self, dtype: torch.dtype) -> Kernel:
        return self.get_or_build_kernel(
            "unpermute_kernel",
            key=dtype,
            build=lambda: self.kernel_map["unpermute_kernel"](
                self.total_tokens, self.top_k, self.hidden_size,
                self.padded_batch_sum, scaling=self._routed_scaling_factor,
                dtype=dtype,
            ),
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"unpermute_kernel": MoeUnpermuteKernel}

    def _infer_output_shapes(
        self,
        mm2_pad_shape: Tuple[int, ...],
        fwd_idx_shape: Tuple[int, ...],
        topk_weights_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == (total_tokens, hidden_size)``."""
        return {"output": (self.total_tokens, self.hidden_size)}

    def forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            mm2_pad: [padded_batch_sum, H] bf16/fp16 down-proj output (padded layout).
            fwd_idx: [T*K] int32 forward mapping: flat_idx → padded slot.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into; allocated internally
                if omitted.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        if out is None:
            return _moe_unpermute_fwd(
                mm2_pad, fwd_idx, topk_weights, self._instance_key)
        _moe_unpermute_fwd_inplace(
            mm2_pad, fwd_idx, topk_weights, out, self._instance_key)
        # The in-place operator returns None so its result cannot alias an input; the
        # buffer the caller handed over is what this op returns.
        return out

    def _eager_forward(
        self,
        mm2_pad: torch.Tensor,
        fwd_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo
        cannot follow.
        """
        self._validate_dtypes(mm2_pad, fwd_idx, topk_weights)
        self.dtype = mm2_pad.dtype
        return self._get_kernel(mm2_pad.dtype)(mm2_pad, fwd_idx, topk_weights, out=out)


@torch.library.custom_op("top::moe_unpermute_fwd", mutates_args=())
def _moe_unpermute_fwd(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(mm2_pad, fwd_idx, topk_weights)


@_moe_unpermute_fwd.register_fake
def _moe_unpermute_fwd_fake(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(mm2_pad.shape), tuple(fwd_idx.shape), tuple(topk_weights.shape))
    # Manifest dtype: ``same_as(mm2_pad)``.
    return mm2_pad.new_empty(shapes["output"])


@torch.library.custom_op("top::moe_unpermute_fwd_inplace", mutates_args=("out",))
def _moe_unpermute_fwd_inplace(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    instance_key: str,
) -> None:
    get_instance(instance_key)._eager_forward(mm2_pad, fwd_idx, topk_weights, out=out)


# The in-place operator needs no fake body: it returns nothing, and the mutation it
# declares is all the compiler has to know about ``out``.
