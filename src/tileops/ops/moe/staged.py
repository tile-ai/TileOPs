"""Public staged Mixture-of-Experts operator boundaries."""

from __future__ import annotations

from typing import Mapping

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops.op_base import Op
from tileops.utils import get_sm_version

from ..elementwise import SiluAndMulFwdOp
from .contracts import (
    InversePermuteContext,
    MaskedLayoutSpec,
    MaterializedExpertLayout,
    MGroupedLayoutSpec,
    NoScaleComputeSpec,
    PrePermuteOutput,
    RoutingEpilogueSpec,
)

__all__ = [
    "MoeExpertMLPFwdOp",
    "MoeGroupedGemmFwdOp",
    "MoePostPermuteFwdOp",
    "MoePrePermuteFwdOp",
]


def _same_device(named_tensors: Mapping[str, torch.Tensor]) -> torch.device:
    devices = {tensor.device for tensor in named_tensors.values()}
    if len(devices) != 1:
        detail = ", ".join(f"{name}={tensor.device}" for name, tensor in named_tensors.items())
        raise ValueError(f"all tensors must share one device; got {detail}")
    return next(iter(devices))


class _SpecOnlyStagedOp(Op):
    """Common behavior for a staged boundary with no shipped implementation yet."""

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {}

    def _infer_output_shapes(self, **shape_kwargs: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError("staged output shape depends on resolved layout metadata")


class MoePrePermuteFwdOp(_SpecOnlyStagedOp):
    """Materialize token-order activations into an expert-compute layout."""

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        num_experts: int,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.layout = layout
        self.num_experts = num_experts
        self.target = target
        self.dispatch_kernel(kernel_map)

    def make_call(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor) -> PrePermuteCall:
        """Validate inputs and construct the immutable selection record."""
        device = _same_device({"hidden_states": hidden_states, "topk_ids": topk_ids})
        if device.type != "cuda":
            raise ValueError("staged pre-permute currently requires CUDA tensors")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if topk_ids.ndim != 2 or topk_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("topk_ids must have shape [tokens, top_k]")
        if topk_ids.shape[1] <= 0:
            raise ValueError("top_k must be positive")
        if topk_ids.dtype is not torch.int32:
            raise TypeError("topk_ids must have dtype torch.int32")
        if not hidden_states.is_contiguous() or not topk_ids.is_contiguous():
            raise ValueError("hidden_states and topk_ids must be contiguous")
        if hidden_states.dtype is not torch.bfloat16:
            raise TypeError("the current staged pre-permute contract accepts BF16 only")
        return PrePermuteCall(
            arch=get_sm_version(device.index),
            layout=self.layout,
            device_type=hidden_states.device.type,
            input_dtype=hidden_states.dtype,
            num_experts=self.num_experts,
            num_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            top_k=topk_ids.shape[1],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> PrePermuteOutput:
        call = self.make_call(hidden_states, topk_ids)
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(hidden_states, topk_ids),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(hidden_states, topk_ids, out=out)


class MoeGroupedGemmFwdOp(_SpecOnlyStagedOp):
    """Independently callable typed M-grouped GEMM boundary."""

    def __init__(
        self,
        compute: NoScaleComputeSpec | None = None,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        compute = NoScaleComputeSpec() if compute is None else compute
        if not isinstance(compute, NoScaleComputeSpec):
            raise TypeError("the current public manifest supports only NoScaleComputeSpec")
        self.compute = compute
        self.target = target
        self.dispatch_kernel(kernel_map)

    def make_call(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        scales: object | None = None,
        out: torch.Tensor | None = None,
    ) -> MGroupedGemmCall:
        """Validate operands/spec/layout and construct the selection record."""
        tensors = {"a": a, "b": b}
        if out is not None:
            tensors["out"] = out
        device = _same_device(tensors)
        if device.type != "cuda":
            raise ValueError("staged grouped GEMM currently requires CUDA tensors")
        expert_layout.validate_structure(a)
        if b.ndim != 3 or b.shape[0] != expert_layout.num_experts:
            raise ValueError("b must have shape [num_experts, n, k]")
        if not b.is_contiguous():
            raise ValueError("b must be contiguous")
        if a.shape[-1] != b.shape[-1]:
            raise ValueError("a and b must have the same reduction dimension")
        arch = get_sm_version(device.index)
        if scales is not None:
            raise ValueError("NoScaleComputeSpec forbids scales")
        if arch != 90 or a.dtype is not torch.bfloat16 or b.dtype is not torch.bfloat16:
            raise ValueError("NoScale currently supports only SM90 BF16 operands")
        output_shape = (*a.shape[:-1], b.shape[1])
        if out is not None and not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out is not None and (
            tuple(out.shape) != output_shape or out.dtype != self.compute.output_dtype
        ):
            raise ValueError("out shape and dtype must match the resolved grouped-GEMM output")
        return MGroupedGemmCall(
            arch=arch,
            layout_key=expert_layout.selection_key,
            max_m=expert_layout.max_m,
            device_type=a.device.type,
            input_dtype=a.dtype,
            weight_dtype=b.dtype,
            output_dtype=self.compute.output_dtype,
            materialized_rows=expert_layout.materialized_rows,
            num_experts=expert_layout.num_experts,
            n=b.shape[1],
            k=b.shape[2],
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        scales: object | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call = self.make_call(a, b, expert_layout, scales, out)
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(a, b),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(a, b, expert_layout, scales=scales, out=out)


class MoeExpertMLPFwdOp(_SpecOnlyStagedOp):
    """Compose two typed grouped GEMMs around a gated activation."""

    def __init__(
        self,
        activation: str = "silu_and_mul",
        *,
        compute: NoScaleComputeSpec | None = None,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        if activation != "silu_and_mul":
            raise ValueError("the staged Expert MLP currently supports only silu_and_mul")
        self.activation = activation
        self.compute = NoScaleComputeSpec() if compute is None else compute
        self.target = target
        self.dispatch_kernel(kernel_map)
        overrides = self.forwarded_overrides()
        grouped_overrides = (
            {key: value for key, value in overrides.items() if key != "silu_and_mul"}
            if overrides
            else None
        )
        self.gate_up = MoeGroupedGemmFwdOp(
            self.compute, kernel_map=grouped_overrides, target=target
        )
        self.activation_op = SiluAndMulFwdOp(kernel_map=overrides)
        self.activation_op.target = target
        self.down = MoeGroupedGemmFwdOp(self.compute, kernel_map=grouped_overrides, target=target)

    def kernel_delegates(self) -> tuple[Op, Op, Op]:
        return self.gate_up, self.activation_op, self.down

    def make_calls(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        out: torch.Tensor | None = None,
    ) -> tuple[MGroupedGemmCall, MGroupedGemmCall]:
        """Build both GEMM calls while preserving one layout binding."""
        gate_call = self.gate_up.make_call(expert_input, w_gate_up, expert_layout)
        if w_gate_up.shape[1] % 2:
            raise ValueError("w_gate_up output dimension must be even")
        intermediate_shape = (*expert_input.shape[:-1], w_gate_up.shape[1] // 2)
        if w_down.shape[-1] != intermediate_shape[-1]:
            raise ValueError("w_down reduction dimension must match the gated width")
        intermediate = expert_input.new_empty(intermediate_shape)
        down_call = self.down.make_call(intermediate, w_down, expert_layout, out=out)
        return gate_call, down_call

    def forward(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        expert_layout: MaterializedExpertLayout,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.make_calls(expert_input, w_gate_up, w_down, expert_layout, out)
        gate_up = self.gate_up(expert_input, w_gate_up, expert_layout)
        flat_gate_up = gate_up.reshape(-1, gate_up.shape[-1])
        activated = self.activation_op(flat_gate_up)
        activated = activated.reshape(*gate_up.shape[:-1], gate_up.shape[-1] // 2)
        return self.down(activated, w_down, expert_layout, out=out)


class MoePostPermuteFwdOp(_SpecOnlyStagedOp):
    """Restore token order and apply the declared local routing epilogue."""

    def __init__(
        self,
        epilogue: RoutingEpilogueSpec | None = None,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        epilogue = RoutingEpilogueSpec() if epilogue is None else epilogue
        if not isinstance(epilogue, RoutingEpilogueSpec):
            raise TypeError("epilogue must be RoutingEpilogueSpec")
        self.epilogue = epilogue
        self.target = target
        self.dispatch_kernel(kernel_map)

    def make_call(
        self,
        expert_output: torch.Tensor,
        inverse_permute_context: InversePermuteContext,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> PostPermuteCall:
        """Validate invocation-bound state and construct the selection record."""
        tensors = {
            "expert_output": expert_output,
            "inverse_indices": inverse_permute_context.inverse_indices,
            "topk_weights": topk_weights,
        }
        if out is not None:
            tensors["out"] = out
        device = _same_device(tensors)
        if device.type != "cuda":
            raise ValueError("staged post-permute currently requires CUDA tensors")
        context = inverse_permute_context
        if not expert_output.is_contiguous():
            raise ValueError("expert_output must be contiguous")
        if expert_output.ndim not in (2, 3):
            raise ValueError("expert_output must be contiguous rank 2 or masked rank 3")
        if isinstance(context.layout, MaskedLayoutSpec):
            expected = (context.num_experts, context.layout.max_m)
            if expert_output.ndim != 3 or tuple(expert_output.shape[:2]) != expected:
                raise ValueError("masked expert_output leading dimensions do not match context")
        elif expert_output.ndim != 2:
            raise ValueError("contiguous expert_output must be rank 2")
        physical_rows = expert_output.numel() // expert_output.shape[-1]
        if physical_rows != context.materialized_rows:
            raise ValueError("expert_output row count does not match inverse context")
        if tuple(topk_weights.shape) != (context.num_tokens, context.top_k):
            raise ValueError("topk_weights shape does not match inverse context")
        if topk_weights.dtype is not torch.float32:
            raise TypeError("topk_weights must have dtype torch.float32")
        output_shape = (context.num_tokens, expert_output.shape[-1])
        if out is not None and not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out is not None and (
            tuple(out.shape) != output_shape or out.dtype != self.epilogue.output_dtype
        ):
            raise ValueError("out shape and dtype must match the routing epilogue output")
        return PostPermuteCall(
            arch=get_sm_version(device.index),
            layout_key=context.selection_key,
            max_m=context.layout.max_m,
            epilogue=self.epilogue,
            device_type=expert_output.device.type,
            input_dtype=expert_output.dtype,
            routing_weight_dtype=topk_weights.dtype,
            output_dtype=self.epilogue.output_dtype,
            num_experts=context.num_experts,
            materialized_rows=context.materialized_rows,
            num_tokens=context.num_tokens,
            hidden_size=expert_output.shape[-1],
            top_k=context.top_k,
        )

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_permute_context: InversePermuteContext,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call = self.make_call(expert_output, inverse_permute_context, topk_weights, out)
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(expert_output, topk_weights),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(
            expert_output,
            inverse_permute_context,
            topk_weights,
            epilogue=self.epilogue,
            out=out,
        )
