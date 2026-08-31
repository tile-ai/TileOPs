"""Public staged Mixture-of-Experts operator boundaries."""

from __future__ import annotations

from typing import ClassVar, Mapping

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MoePermuteNopadKernel,
    MoeUnpermuteKernel,
)
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops.compile_boundary import get_instance
from tileops.ops.op_base import Op
from tileops.utils import get_sm_version

from ..elementwise import SiluAndMulFwdOp
from .contracts import (
    MaskedLayoutSpec,
    MGroupedLayoutSpec,
    NoScaleComputeSpec,
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


class _TightPhysicalPsumPrePermuteKernel(Kernel):
    """Adapter from the staged contract to the shipped no-pad permute kernel."""

    supported_archs = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: PrePermuteCall) -> bool:
        return call.layout.selection_key == "tight_physical_psum"

    def __init__(self, call: PrePermuteCall) -> None:
        """Build the shipped tight physical-prefix-sum specialization."""
        super().__init__()
        self.call = call
        self.inner = MoePermuteNopadKernel(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.hidden_size,
            call.input_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize expert rows, physical segment ends, and inverse indices."""
        perm_h, true_offsets, true_sizes, _, inverse_indices = self.inner(
            hidden_states, local_expert_ids
        )
        # Physical segment ends are the layout-level representation shared by all
        # staged grouped-GEMM candidates.  Keep device values asynchronous.
        physical_ends = true_offsets + true_sizes
        return perm_h, physical_ends, inverse_indices


class _TightPhysicalPsumPostPermuteKernel(Kernel):
    """Adapter from staged inverse indices to the shipped weighted unpermute kernel."""

    supported_archs = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: PostPermuteCall) -> bool:
        return call.layout_key == "tight_physical_psum" and call.output_dtype == call.input_dtype

    def __init__(self, call: PostPermuteCall) -> None:
        """Build the shipped tight physical-prefix-sum specialization."""
        super().__init__()
        self.call = call
        self.inner = MoeUnpermuteKernel(
            call.num_tokens,
            call.top_k,
            call.hidden_size,
            call.materialized_rows,
            scaling=call.epilogue.routed_scaling_factor,
            dtype=call.input_dtype,
        )

    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        epilogue: RoutingEpilogueSpec,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore rank-grouped order with weighted reduction and optional output reuse."""
        return self.inner(
            expert_output,
            inverse_indices,
            topk_weights,
            out=out,
        )


class _StagedOpBase(Op):
    """Shared defaults for staged boundaries whose concrete class may override them."""

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {}

    def _infer_output_shapes(self, **shape_kwargs: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError("staged output shape depends on materialized layout metadata")

    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        raise NotImplementedError("staged dtype validation requires family-specific call data")

    def eval_roofline(self) -> tuple[int, int]:
        raise NotImplementedError("staged roofline evaluation requires materialized runtime data")


class MoePrePermuteFwdOp(_StagedOpBase):
    """Materialize rank-grouped activations into a local expert-compute layout.

    ``local_expert_ids`` must already be remapped into
    ``[0, num_local_experts)`` by EPDispatch.  This op knows no global expert
    placement and performs no communication.
    """

    compile_op_names: ClassVar[tuple[str, ...]] = ("tileops::moe_pre_permute_fwd",)

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        num_local_experts: int,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure a pre-permute boundary for one layout and expert domain."""
        if num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        self.layout = layout
        self.num_local_experts = num_local_experts
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {"tight_physical_psum": _TightPhysicalPsumPrePermuteKernel}

    def _validate_dtypes(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> None:
        if hidden_states.dtype is not torch.bfloat16:
            raise TypeError("hidden_states must have dtype BF16")
        if local_expert_ids.dtype is not torch.int32:
            raise TypeError("local_expert_ids must have dtype int32")

    def _infer_output_shapes(
        self,
        hidden_states_shape: tuple[int, ...],
        local_expert_ids_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        rows = hidden_states_shape[0] * local_expert_ids_shape[1]
        if isinstance(self.layout, MaskedLayoutSpec):
            expert_input = (
                self.num_local_experts,
                self.layout.max_m,
                hidden_states_shape[1],
            )
        else:
            expert_input = (rows, hidden_states_shape[1])
        layout_key = getattr(self.layout, "selection_key", self.layout)
        metadata_rows = rows if layout_key == "tight_per_row" else self.num_local_experts
        return {
            "expert_input": expert_input,
            "layout_metadata": (metadata_rows,),
            "inverse_indices": (rows,),
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self.input_shapes is None or self.dtype is None:
            raise RuntimeError("eval_roofline requires a prior forward call")
        hidden_shape, ids_shape = self.input_shapes
        tokens, hidden = hidden_shape
        rows = tokens * ids_shape[1]
        output_shapes = self._infer_output_shapes(hidden_shape, ids_shape)
        expert_numel = 1
        for dim in output_shapes["expert_input"]:
            expert_numel *= dim
        metadata_numel = output_shapes["layout_metadata"][0]
        itemsize = self.dtype.itemsize
        nbytes = (tokens * hidden + expert_numel) * itemsize
        nbytes += (ids_shape[0] * ids_shape[1] + rows + metadata_numel) * 4
        return 0, int(nbytes)

    def make_call(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> PrePermuteCall:
        """Validate inputs and construct the immutable selection record."""
        device = _same_device(
            {"hidden_states": hidden_states, "local_expert_ids": local_expert_ids}
        )
        if device.type != "cuda":
            raise ValueError("staged pre-permute currently requires CUDA tensors")
        if hidden_states.ndim != 2:
            raise ValueError("hidden_states must have shape [tokens, hidden_size]")
        if local_expert_ids.ndim != 2 or local_expert_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError("local_expert_ids must have shape [tokens, top_k]")
        if local_expert_ids.shape[1] <= 0:
            raise ValueError("top_k must be positive")
        self._validate_dtypes(hidden_states, local_expert_ids)
        if not hidden_states.is_contiguous() or not local_expert_ids.is_contiguous():
            raise ValueError("hidden_states and local_expert_ids must be contiguous")
        torch._assert_async(
            torch.all((local_expert_ids >= 0) & (local_expert_ids < self.num_local_experts)),
            "local_expert_ids must be in [0, num_local_experts)",
        )
        return PrePermuteCall(
            arch=get_sm_version(device.index),
            layout=self.layout,
            device_type=hidden_states.device.type,
            input_dtype=hidden_states.dtype,
            num_experts=self.num_local_experts,
            num_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            top_k=local_expert_ids.shape[1],
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(expert_input, layout_metadata, inverse_indices)``."""
        return _moe_pre_permute_fwd(hidden_states, local_expert_ids, self._instance_key)

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        call = self.make_call(hidden_states, local_expert_ids)
        self.dtype = hidden_states.dtype
        self.input_shapes = [tuple(hidden_states.shape), tuple(local_expert_ids.shape)]
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(hidden_states, local_expert_ids),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(hidden_states, local_expert_ids)


class MoeGroupedGemmFwdOp(_StagedOpBase):
    """Independently callable typed M-grouped GEMM boundary."""

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        compute: NoScaleComputeSpec | None = None,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure the current BF16/NoScale M-grouped GEMM semantic."""
        self.layout = layout
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
        layout_metadata: torch.Tensor,
        scales: object | None = None,
        out: torch.Tensor | None = None,
    ) -> MGroupedGemmCall:
        """Validate operands/spec/layout and construct the selection record."""
        tensors = {"a": a, "b": b, "layout_metadata": layout_metadata}
        if out is not None:
            tensors["out"] = out
        device = _same_device(tensors)
        if device.type != "cuda":
            raise ValueError("staged grouped GEMM currently requires CUDA tensors")
        if b.ndim != 3:
            raise ValueError("b must have shape [num_experts, n, k]")
        num_experts = b.shape[0]
        if layout_metadata.dtype is not torch.int32 or not layout_metadata.is_contiguous():
            raise TypeError("layout_metadata must be contiguous int32")
        if isinstance(self.layout, MaskedLayoutSpec):
            if a.ndim != 3 or tuple(a.shape[:2]) != (num_experts, self.layout.max_m):
                raise ValueError("masked a must have shape [num_experts, max_m, k]")
            expected_metadata = num_experts
        else:
            if a.ndim != 2:
                raise ValueError("contiguous a must have shape [rows, k]")
            expected_metadata = (
                a.shape[0] if self.layout.selection_key == "tight_per_row" else num_experts
            )
        if layout_metadata.ndim != 1 or layout_metadata.shape[0] != expected_metadata:
            raise ValueError("layout_metadata shape does not match the selected layout")
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
            layout_key=self.layout.selection_key,
            max_m=self.layout.max_m,
            device_type=a.device.type,
            input_dtype=a.dtype,
            weight_dtype=b.dtype,
            output_dtype=self.compute.output_dtype,
            materialized_rows=a.numel() // a.shape[-1],
            num_experts=num_experts,
            n=b.shape[1],
            k=b.shape[2],
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        scales: object | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one grouped GEMM using the supplied materialized expert layout."""
        call = self.make_call(a, b, layout_metadata, scales, out)
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(a, b, layout_metadata),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(a, b, layout_metadata, scales=scales, out=out)


class MoeExpertMLPFwdOp(_StagedOpBase):
    """Compose two typed grouped GEMMs around a gated activation."""

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        activation: str = "silu_and_mul",
        *,
        compute: NoScaleComputeSpec | None = None,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure two grouped GEMMs around the selected gated activation."""
        if activation != "silu_and_mul":
            raise ValueError("the staged Expert MLP currently supports only silu_and_mul")
        self.activation = activation
        self.layout = layout
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
            layout, self.compute, kernel_map=grouped_overrides, target=target
        )
        self.activation_op = SiluAndMulFwdOp(kernel_map=overrides)
        self.activation_op.target = target
        self.down = MoeGroupedGemmFwdOp(
            layout, self.compute, kernel_map=grouped_overrides, target=target
        )

    def kernel_delegates(self) -> tuple[Op, Op, Op]:
        return self.gate_up, self.activation_op, self.down

    def make_calls(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[MGroupedGemmCall, MGroupedGemmCall]:
        """Build both GEMM calls while preserving one layout binding."""
        gate_call = self.gate_up.make_call(expert_input, w_gate_up, layout_metadata)
        if w_gate_up.shape[1] % 2:
            raise ValueError("w_gate_up output dimension must be even")
        intermediate_shape = (*expert_input.shape[:-1], w_gate_up.shape[1] // 2)
        if w_down.shape[-1] != intermediate_shape[-1]:
            raise ValueError("w_down reduction dimension must match the gated width")
        intermediate = expert_input.new_empty(intermediate_shape)
        down_call = self.down.make_call(intermediate, w_down, layout_metadata, out=out)
        return gate_call, down_call

    def forward(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run gate/up GEMM, gated activation, and down GEMM on one layout."""
        self.make_calls(expert_input, w_gate_up, w_down, layout_metadata, out)
        gate_up = self.gate_up(expert_input, w_gate_up, layout_metadata)
        flat_gate_up = gate_up.reshape(-1, gate_up.shape[-1])
        activated = self.activation_op(flat_gate_up)
        activated = activated.reshape(*gate_up.shape[:-1], gate_up.shape[-1] // 2)
        return self.down(activated, w_down, layout_metadata, out=out)


class MoePostPermuteFwdOp(_StagedOpBase):
    """Restore rank-grouped order and apply the local routing epilogue.

    Cross-rank return/scatter belongs to EPCombine; without EP, rank-grouped
    order is the caller's original token order.
    """

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        epilogue: RoutingEpilogueSpec | None = None,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure inverse permutation and the exactly-once routing epilogue."""
        self.layout = layout
        epilogue = RoutingEpilogueSpec() if epilogue is None else epilogue
        if not isinstance(epilogue, RoutingEpilogueSpec):
            raise TypeError("epilogue must be RoutingEpilogueSpec")
        self.epilogue = epilogue
        self.target = target
        self.dispatch_kernel(kernel_map)

    compile_op_names: ClassVar[tuple[str, ...]] = (
        "tileops::moe_post_permute_fwd",
        "tileops::moe_post_permute_fwd_inplace",
    )

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {"tight_physical_psum": _TightPhysicalPsumPostPermuteKernel}

    def _validate_dtypes(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> None:
        if expert_output.dtype is not torch.bfloat16:
            raise TypeError("expert_output must have dtype BF16")
        if topk_weights.dtype is not torch.float32:
            raise TypeError("topk_weights must have dtype float32")
        if inverse_indices.dtype is not torch.int32:
            raise TypeError("inverse_indices must have dtype int32")

    def _infer_output_shapes(
        self,
        expert_output_shape: tuple[int, ...],
        topk_weights_shape: tuple[int, ...],
        inverse_indices_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        return {"output": (topk_weights_shape[0], expert_output_shape[-1])}

    def eval_roofline(self) -> tuple[int, int]:
        if self.input_shapes is None or self.dtype is None:
            raise RuntimeError("eval_roofline requires a prior forward call")
        expert_shape, weights_shape, inverse_shape = self.input_shapes
        rows = 1
        for dim in expert_shape[:-1]:
            rows *= dim
        hidden = expert_shape[-1]
        tokens, top_k = weights_shape
        itemsize = self.dtype.itemsize
        flops = 2 * tokens * top_k * hidden
        nbytes = (rows * hidden + tokens * hidden) * itemsize
        nbytes += (weights_shape[0] * weights_shape[1] + inverse_shape[0]) * 4
        return int(flops), int(nbytes)

    def make_call(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> PostPermuteCall:
        """Validate invocation-bound state and construct the selection record."""
        tensors = {
            "expert_output": expert_output,
            "inverse_indices": inverse_indices,
            "topk_weights": topk_weights,
        }
        if out is not None:
            tensors["out"] = out
        device = _same_device(tensors)
        if device.type != "cuda":
            raise ValueError("staged post-permute currently requires CUDA tensors")
        if not expert_output.is_contiguous():
            raise ValueError("expert_output must be contiguous")
        if expert_output.ndim not in (2, 3):
            raise ValueError("expert_output must be contiguous rank 2 or masked rank 3")
        if isinstance(self.layout, MaskedLayoutSpec):
            if expert_output.ndim != 3 or expert_output.shape[1] != self.layout.max_m:
                raise ValueError("masked expert_output must have shape [experts, max_m, hidden]")
        elif expert_output.ndim != 2:
            raise ValueError("contiguous expert_output must be rank 2")
        physical_rows = expert_output.numel() // expert_output.shape[-1]
        if inverse_indices.ndim != 1 or inverse_indices.numel() != topk_weights.numel():
            raise ValueError("inverse_indices must have one entry per routing weight")
        self._validate_dtypes(expert_output, topk_weights, inverse_indices)
        torch._assert_async(
            torch.all((inverse_indices >= -1) & (inverse_indices < physical_rows)),
            "inverse_indices must be -1 or index a materialized expert row",
        )
        if topk_weights.ndim != 2:
            raise ValueError("topk_weights must have shape [tokens, top_k]")
        output_shape = (topk_weights.shape[0], expert_output.shape[-1])
        if out is not None and not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out is not None and (
            tuple(out.shape) != output_shape or out.dtype != self.epilogue.output_dtype
        ):
            raise ValueError("out shape and dtype must match the routing epilogue output")
        return PostPermuteCall(
            arch=get_sm_version(device.index),
            layout_key=self.layout.selection_key,
            max_m=self.layout.max_m,
            epilogue=self.epilogue,
            device_type=expert_output.device.type,
            input_dtype=expert_output.dtype,
            routing_weight_dtype=topk_weights.dtype,
            output_dtype=self.epilogue.output_dtype,
            num_experts=expert_output.shape[0] if isinstance(self.layout, MaskedLayoutSpec) else 0,
            materialized_rows=physical_rows,
            num_tokens=topk_weights.shape[0],
            hidden_size=expert_output.shape[-1],
            top_k=topk_weights.shape[1],
        )

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore token order, apply routing weights, reduce top-k, and cast."""
        if out is None:
            return _moe_post_permute_fwd(
                expert_output, inverse_indices, topk_weights, self._instance_key
            )
        _moe_post_permute_fwd_inplace(
            expert_output, inverse_indices, topk_weights, out, self._instance_key
        )
        return out

    def _eager_forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call = self.make_call(expert_output, inverse_indices, topk_weights, out)
        self.dtype = expert_output.dtype
        self.input_shapes = [
            tuple(expert_output.shape),
            tuple(topk_weights.shape),
            tuple(inverse_indices.shape),
        ]
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(expert_output, topk_weights, inverse_indices),
            key=call,
            build=lambda: self.kernel_map[name](call),
        )
        return kernel(
            expert_output,
            inverse_indices,
            topk_weights,
            epilogue=self.epilogue,
            out=out,
        )


@torch.library.custom_op("tileops::moe_pre_permute_fwd", mutates_args=())
def _moe_pre_permute_fwd(
    hidden_states: torch.Tensor,
    local_expert_ids: torch.Tensor,
    instance_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_instance(instance_key)._eager_forward(hidden_states, local_expert_ids)


@_moe_pre_permute_fwd.register_fake
def _moe_pre_permute_fwd_fake(
    hidden_states: torch.Tensor,
    local_expert_ids: torch.Tensor,
    instance_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(hidden_states.shape), tuple(local_expert_ids.shape))
    return (
        hidden_states.new_empty(shapes["expert_input"]),
        torch.empty(shapes["layout_metadata"], dtype=torch.int32, device=hidden_states.device),
        torch.empty(shapes["inverse_indices"], dtype=torch.int32, device=hidden_states.device),
    )


@torch.library.custom_op("tileops::moe_post_permute_fwd", mutates_args=())
def _moe_post_permute_fwd(
    expert_output: torch.Tensor,
    inverse_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(expert_output, inverse_indices, topk_weights)


@_moe_post_permute_fwd.register_fake
def _moe_post_permute_fwd_fake(
    expert_output: torch.Tensor,
    inverse_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return expert_output.new_empty((topk_weights.shape[0], expert_output.shape[-1]))


@torch.library.custom_op("tileops::moe_post_permute_fwd_inplace", mutates_args=("out",))
def _moe_post_permute_fwd_inplace(
    expert_output: torch.Tensor,
    inverse_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    instance_key: str,
) -> None:
    get_instance(instance_key)._eager_forward(expert_output, inverse_indices, topk_weights, out=out)
