"""Public staged Mixture-of-Experts operator boundaries."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Mapping

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import MoePrePermuteContiguousKernel, MoeUnpermuteKernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops.compile_boundary import get_instance
from tileops.ops.op_base import Op
from tileops.perf.formulas import moe_expert_mlp_roofline, moe_grouped_gemm_roofline
from tileops.utils import get_sm_version, is_h200

from .contracts import (
    ContiguousLayoutSpec,
    ContiguousMetadata,
    ContiguousPacking,
    MaskedLayoutSpec,
    MGroupedLayoutSpec,
    RoutingEpilogueSpec,
    layout_value_guard,
)

__all__ = [
    "MoeExpertMLPFwdOp",
    "MoeGroupedGemmFwdOp",
    "MoePostPermuteFwdOp",
    "MoePrePermuteFwdOp",
]


# Gated activations a grouped GEMM may fuse into its epilogue.
GATED_ACTIVATIONS = ("silu_and_mul", "gelu_and_mul")


def _same_device(named_tensors: Mapping[str, torch.Tensor]) -> torch.device:
    devices = {tensor.device for tensor in named_tensors.values()}
    if len(devices) != 1:
        detail = ", ".join(f"{name}={tensor.device}" for name, tensor in named_tensors.items())
        raise ValueError(f"all tensors must share one device; got {detail}")
    return next(iter(devices))


class _ContiguousPostPermuteKernel(Kernel):
    """Adapt contiguous staged indices to the shipped weighted unpermute kernel."""

    supported_archs = [80, 86, 89, 90]

    @classmethod
    def applies(cls, call: PostPermuteCall) -> bool:
        return (
            call.layout_key in ("tight_physical_psum", "aligned_per_row")
            and call.input_dtype in (torch.bfloat16, torch.float16)
            and call.output_dtype == call.input_dtype
        )

    def __init__(self, call: PostPermuteCall) -> None:
        """Build the weighted no-pad inverse specialization selected by ``call``."""
        super().__init__()
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
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Restore token order and apply the configured routing epilogue."""
        return self.inner(expert_output, inverse_indices, topk_weights, out=out)


class _StagedOpBase(Op):
    """Common behavior of the staged boundary ops: no default candidates, dtypes checked per call."""

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {}

    def _validate_dtypes(self, *args: torch.Tensor) -> None:
        raise NotImplementedError("staged dtype validation requires family-specific call data")


class MoePrePermuteFwdOp(_StagedOpBase):
    """Materialize rank-grouped activations into a local expert layout.

    ``local_expert_ids`` must already be in ``[0, num_local_experts)``.
    Global placement and communication belong to EPDispatch.
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
        self.layout = _check_layout(layout)
        self.num_local_experts = num_local_experts
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {"contiguous": MoePrePermuteContiguousKernel}

    def _infer_output_shapes(
        self,
        hidden_states_shape: tuple[int, ...],
        local_expert_ids_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        rows = hidden_states_shape[0] * local_expert_ids_shape[1]
        # The manifest validator's parity probe builds the op without __init__ and
        # binds a placeholder for ``layout``; getattr resolves it to the tight shapes.
        layout = self.layout
        if isinstance(layout, MaskedLayoutSpec):
            expert_input = (self.num_local_experts, layout.max_m, hidden_states_shape[1])
            metadata_rows = self.num_local_experts
        else:
            per_row = getattr(layout, "metadata_kind", None) is ContiguousMetadata.PER_ROW
            capacity = rows
            if per_row and getattr(layout, "packing", None) is ContiguousPacking.ALIGNED:
                capacity += self.num_local_experts * (layout.alignment - 1)
            expert_input = (capacity, hidden_states_shape[1])
            metadata_rows = capacity if per_row else self.num_local_experts
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
        nbytes = (tokens * hidden + expert_numel) * self.dtype.itemsize
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
        if local_expert_ids.dtype is not torch.int32:
            raise TypeError("local_expert_ids must have dtype torch.int32")
        if not hidden_states.is_contiguous() or not local_expert_ids.is_contiguous():
            raise ValueError("hidden_states and local_expert_ids must be contiguous")
        if hidden_states.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("the current staged pre-permute contract accepts BF16 or FP16 only")
        return PrePermuteCall(
            arch=get_sm_version(device.index),
            h200=is_h200(device.index),
            layout=self.layout,
            device_type=hidden_states.device.type,
            input_dtype=hidden_states.dtype,
            num_experts=self.num_local_experts,
            num_tokens=hidden_states.shape[0],
            hidden_size=hidden_states.shape[1],
            top_k=local_expert_ids.shape[1],
            routing_input_kind="local_expert_ids",
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


def _check_layout(layout: object) -> MGroupedLayoutSpec:
    if not isinstance(layout, (ContiguousLayoutSpec, MaskedLayoutSpec)):
        raise TypeError("layout must be ContiguousLayoutSpec or MaskedLayoutSpec")
    return layout


def _rows_of(layout: MGroupedLayoutSpec, a_shape: tuple[int, ...]) -> int:
    """Materialized rows of ``a``: ``M`` for contiguous, ``E * max_m`` for masked."""
    if isinstance(layout, MaskedLayoutSpec):
        return a_shape[0] * a_shape[1]
    return a_shape[0]


class MoeGroupedGemmFwdOp(_StagedOpBase):
    """M-grouped GEMM over expert-materialized rows: ``out[rows of g] = a[rows of g] @ b[g]^T``.

    ``layout`` fixes how the rows of ``a`` are organised by expert and what the
    ``layout_metadata`` tensor means; ``E`` is read off ``b``, ``M``/``N``/``K`` off
    the operands. Accumulation is fp32; the output is written in the operand dtype
    unless ``out_dtype`` asks for fp32.

    With ``activation`` set, ``b`` stacks the gate and up projections along ``N``
    (``[E, 2 * ffn, K]``) and ``out`` is ``act(gate) * up`` with ``ffn`` columns: the
    gated activation is fused into the GEMM's epilogue and the ``[.., 2 * ffn]``
    intermediate is never written.

    Per layout the operands are (``trans``-free, ``b`` is ``[E, N, K]``):

    | layout                      | ``a``             | ``layout_metadata``          | ``out``           |
    | --------------------------- | ----------------- | ---------------------------- | ----------------- |
    | contiguous · physical_psum  | ``[M, K]``        | ``[E]`` segment end rows     | ``[M, N]``        |
    | contiguous · per_row        | ``[M, K]``        | ``[M]`` expert id per row    | ``[M, N]``        |
    | masked                      | ``[E, max_m, K]`` | ``[E]`` valid rows per expert| ``[E, max_m, N]`` |

    Rows an aligned layout pads with, and a masked slab's rows past its valid
    count, hold unspecified values in ``out``. The metadata's value-level
    invariants (ordering, segment ends, ranges) are not checked here — doing so
    would synchronise — but ``layout_guard`` returns them as an asynchronous
    tensor for tests and benchmarks.
    """

    compile_op_names: ClassVar[tuple[str, ...]] = (
        "tileops::moe_grouped_gemm_fwd",
        "tileops::moe_grouped_gemm_fwd_inplace",
    )

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        *,
        activation: str | None = None,
        out_dtype: torch.dtype | None = None,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Fix the expert layout, the fused activation and the output dtype policy.

        Args:
            layout: Manifest ``params.layout``; how ``a``'s rows are grouped by expert.
            activation: Manifest ``params.activation``; ``None`` for a plain GEMM, or a
                gated activation (``"silu_and_mul"``, ``"gelu_and_mul"``) fused into the
                epilogue over a gate||up ``b``, which halves the output width.
            out_dtype: Manifest ``params.out_dtype``; ``None`` writes the operand dtype,
                ``torch.float32`` keeps the fp32 accumulator.
            kernel_map: Optional kernel override dict.
            target: Which backend serves this instance; detected from the tensors when
                ``None``.
        """
        self.layout = _check_layout(layout)
        if activation is not None and activation not in GATED_ACTIVATIONS:
            raise ValueError(f"activation must be None or one of {GATED_ACTIVATIONS}")
        self.activation = activation
        if out_dtype is not None and out_dtype is not torch.float32:
            raise ValueError("out_dtype must be None (operand dtype) or torch.float32")
        self.out_dtype = out_dtype
        self.target = target
        self.dispatch_kernel(kernel_map)

    def resolve_output_dtype(self, input_dtype: torch.dtype) -> torch.dtype:
        """The dtype ``out`` is written in for operands of ``input_dtype``."""
        return input_dtype if self.out_dtype is None else self.out_dtype

    def _infer_output_shapes(
        self,
        a_shape: tuple[int, ...],
        b_shape: tuple[int, ...],
        layout_metadata_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        # The manifest validator's parity probe builds the op without __init__ and
        # binds a placeholder for ``activation``; anything but None means fused.
        n = b_shape[1]
        if getattr(self, "activation", None) is not None:
            n //= 2
        return {"output": (*tuple(a_shape)[:-1], n)}

    def eval_roofline(self) -> tuple[int, int]:
        # What codegen emits for ``roofline.func``; a spec-only entry gets no codegen.
        return moe_grouped_gemm_roofline(self)

    def make_call(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> MGroupedGemmCall:
        """Validate the operands against the layout and build the selection record."""
        tensors = {"a": a, "b": b, "layout_metadata": layout_metadata}
        if out is not None:
            tensors["out"] = out
        device = _same_device(tensors)
        if device.type != "cuda":
            raise ValueError("staged grouped GEMM currently requires CUDA tensors")
        if a.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("the staged grouped-GEMM contract accepts BF16 or FP16 operands")
        if b.dtype is not a.dtype:
            raise TypeError("a and b must share one dtype")
        if layout_metadata.dtype is not torch.int32:
            raise TypeError("layout_metadata must have dtype torch.int32")
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if b.ndim != 3:
            raise ValueError("b must have shape [num_experts, n, k]")
        num_experts, n, k = b.shape
        layout = self.layout
        if isinstance(layout, MaskedLayoutSpec):
            if a.ndim != 3 or a.shape[0] != num_experts or a.shape[1] != layout.max_m:
                raise ValueError(
                    f"masked a must have shape [{num_experts}, {layout.max_m}, k] to match "
                    f"b and the layout's max_m"
                )
        else:
            if a.ndim != 2:
                raise ValueError("contiguous a must have shape [rows, k]")
            if layout.packing is ContiguousPacking.ALIGNED and a.shape[0] % layout.alignment:
                raise ValueError(
                    f"aligned rows ({a.shape[0]}) must be a multiple of the layout's "
                    f"alignment ({layout.alignment})"
                )
        if a.shape[-1] != k:
            raise ValueError("a and b must have the same reduction dimension")
        if self.activation is not None and n % 2:
            raise ValueError(
                f"a fused gated activation splits b's N={n} into gate and up halves: N must be even"
            )
        rows = _rows_of(layout, tuple(a.shape))
        expected_meta = layout.metadata_length(rows=rows, num_experts=num_experts)
        if layout_metadata.ndim != 1 or layout_metadata.shape[0] != expected_meta:
            raise ValueError(
                f"layout_metadata must have shape ({expected_meta},) for this layout, "
                f"got {tuple(layout_metadata.shape)}"
            )
        cd_dtype = self.resolve_output_dtype(a.dtype)
        output_shape = self._infer_output_shapes(
            tuple(a.shape), tuple(b.shape), tuple(layout_metadata.shape)
        )["output"]
        if out is not None and (tuple(out.shape) != output_shape or out.dtype != cd_dtype):
            raise ValueError(
                f"out must be a {list(output_shape)} tensor of dtype {cd_dtype}, "
                f"got {list(out.shape)} {out.dtype}"
            )
        packing = None if isinstance(layout, MaskedLayoutSpec) else layout.packing.value
        metadata_kind = None if isinstance(layout, MaskedLayoutSpec) else layout.metadata_kind.value
        return MGroupedGemmCall(
            arch=get_sm_version(device.index),
            h200=is_h200(device.index),
            kind=layout.kind,
            packing=packing,
            metadata_kind=metadata_kind,
            alignment=getattr(layout, "alignment", 1),
            max_m=layout.max_m,
            activation=self.activation,
            ab_dtype=a.dtype,
            cd_dtype=cd_dtype,
            num_groups=num_experts,
            m=rows,
            n=n,
            k=k,
        )

    def layout_guard(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Asynchronous bool: ``layout_metadata``'s values satisfy the layout for ``a``/``b``.

        The metadata's shape is checked here on the host, its values on the device;
        never a host sync. Consume with ``torch._assert_async`` or ``.item()`` in
        tests and benchmarks, never in ``forward``.

        Raises:
            ValueError: ``layout_metadata`` is not the int32 vector the layout expects.
        """
        rows = _rows_of(self.layout, tuple(a.shape))
        expected = self.layout.metadata_length(rows=rows, num_experts=b.shape[0])
        if layout_metadata.dtype is not torch.int32 or tuple(layout_metadata.shape) != (expected,):
            raise ValueError(
                f"layout_metadata must be an int32 vector of length {expected}, "
                f"got {layout_metadata.dtype} {tuple(layout_metadata.shape)}"
            )
        return layout_value_guard(
            self.layout,
            layout_metadata,
            rows=rows,
            num_experts=b.shape[0],
        )

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one GEMM per expert over the rows the layout assigns it.

        Args:
            a: Expert-materialized activations, ``[M, K]`` or ``[E, max_m, K]`` per the layout.
            b: Per-expert weights, ``[E, N, K]``.
            layout_metadata: ``int32`` metadata whose shape and meaning the layout fixes.
            out: Optional preallocated output in the resolved output dtype.

        Returns:
            ``[M, N]`` or ``[E, max_m, N]`` in the operand dtype, or fp32 when asked for.
        """
        if out is None:
            return _moe_grouped_gemm_fwd(a, b, layout_metadata, self._instance_key)
        _moe_grouped_gemm_fwd_inplace(a, b, layout_metadata, out, self._instance_key)
        return out

    def _eager_forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        call = self.make_call(a, b, layout_metadata, out)
        self.dtype = a.dtype
        self.input_shapes = [tuple(a.shape), tuple(b.shape), tuple(layout_metadata.shape)]
        name = self.select_kernel_key(tuple((self.kernel_map or {}).keys()), call)
        # ``m`` is a fact of the call, not of the built kernel. The builder is handed
        # the record the cache is keyed on, so it cannot specialize on a row count.
        build_call = dataclasses.replace(call, m=0)
        kernel = self.get_or_build_kernel(
            name,
            inputs=(a, b, layout_metadata),
            key=build_call,
            build=lambda: self.kernel_map[name](build_call),
        )
        return kernel(a, b, layout_metadata, out=out)


class MoeExpertMLPFwdOp(_StagedOpBase):
    """Two grouped GEMMs on one expert layout, the gated activation fused into the first.

    ``out = (act(expert_input @ w_gate_up[g]^T)) @ w_down[g]^T`` per expert ``g``, where
    ``act`` is the gated activation: ``w_gate_up`` stacks the gate and up projections
    along ``N`` and the gate_up GEMM's epilogue applies the activation and halves it,
    so the ``[.., 2 * ffn]`` intermediate is never written. A composite: it registers
    no operator of its own, its graph is its two leaves'.
    """

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        activation: str = "silu_and_mul",
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure two grouped GEMMs on ``layout``, the first fusing the gated activation.

        Args:
            layout: Manifest ``params.layout``; shared by both GEMMs and the metadata.
            activation: Manifest ``params.activation``; ``"silu_and_mul"`` or
                ``"gelu_and_mul"``, fused into the gate_up GEMM's epilogue.
            kernel_map: Optional overrides, forwarded to both GEMMs.
            target: Which backend serves the delegates.
        """
        if activation not in GATED_ACTIVATIONS:
            raise ValueError(f"activation must be one of {GATED_ACTIVATIONS}, got {activation!r}")
        self.layout = _check_layout(layout)
        self.activation = activation
        self.target = target
        self.dispatch_kernel(kernel_map)
        overrides = self.forwarded_overrides() or None
        self.gate_up = MoeGroupedGemmFwdOp(
            layout, activation=activation, kernel_map=overrides, target=target
        )
        self.down = MoeGroupedGemmFwdOp(layout, kernel_map=overrides, target=target)

    def kernel_delegates(self) -> tuple[Op, Op]:
        return self.gate_up, self.down

    def _infer_output_shapes(
        self,
        expert_input_shape: tuple[int, ...],
        w_gate_up_shape: tuple[int, ...],
        w_down_shape: tuple[int, ...],
        layout_metadata_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        return {"output": (*tuple(expert_input_shape)[:-1], w_down_shape[1])}

    def eval_roofline(self) -> tuple[int, int]:
        # What codegen emits for ``roofline.func``; a spec-only entry gets no codegen.
        return moe_expert_mlp_roofline(self)

    @staticmethod
    def _check_widths(w_gate_up: torch.Tensor, w_down: torch.Tensor) -> None:
        if w_gate_up.ndim != 3 or w_down.ndim != 3:
            raise ValueError("w_gate_up and w_down must have shape [num_experts, n, k]")
        if w_gate_up.shape[1] % 2:
            raise ValueError("w_gate_up output dimension must be even")
        if w_down.shape[-1] != w_gate_up.shape[1] // 2:
            raise ValueError("w_down reduction dimension must match the gated width")
        if w_down.shape[0] != w_gate_up.shape[0]:
            raise ValueError("w_gate_up and w_down must hold the same experts")

    def forward(
        self,
        expert_input: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
        layout_metadata: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the gate/up GEMM with its fused activation, then the down GEMM.

        Args:
            expert_input: ``[M, hidden]`` or ``[E, max_m, hidden]`` per the layout.
            w_gate_up: ``[E, 2 * ffn, hidden]``, gate and up stacked along the output axis.
            w_down: ``[E, hidden, ffn]``.
            layout_metadata: ``int32`` metadata the layout fixes; shared by both GEMMs.
            out: Optional preallocated ``[.., hidden]`` output in the operand dtype.

        Returns:
            ``[M, hidden]`` or ``[E, max_m, hidden]`` in the operand dtype.
        """
        self._check_widths(w_gate_up, w_down)
        self.dtype = expert_input.dtype
        self.input_shapes = [
            tuple(expert_input.shape),
            tuple(w_gate_up.shape),
            tuple(w_down.shape),
            tuple(layout_metadata.shape),
        ]
        activated = self.gate_up(expert_input, w_gate_up, layout_metadata)
        return self.down(activated, w_down, layout_metadata, out=out)


class MoePostPermuteFwdOp(_StagedOpBase):
    """Restore token order and apply the declared local routing epilogue."""

    compile_op_names: ClassVar[tuple[str, ...]] = (
        "tileops::moe_post_permute_fwd",
        "tileops::moe_post_permute_fwd_inplace",
    )

    def __init__(
        self,
        layout: MGroupedLayoutSpec,
        epilogue: RoutingEpilogueSpec | None = None,
        *,
        kernel_map: dict[str, Kernel] | None = None,
        target: object = None,
    ) -> None:
        """Configure inverse permutation and the exactly-once routing epilogue."""
        epilogue = RoutingEpilogueSpec() if epilogue is None else epilogue
        if not isinstance(epilogue, RoutingEpilogueSpec):
            raise TypeError("epilogue must be RoutingEpilogueSpec")
        self.layout = _check_layout(layout)
        self.epilogue = epilogue
        self.target = target
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> dict[str, Kernel]:
        return {"contiguous": _ContiguousPostPermuteKernel}

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
        flops = 2 * tokens * top_k * hidden
        nbytes = (rows * hidden + tokens * hidden) * self.dtype.itemsize
        nbytes += (weights_shape[0] * weights_shape[1] + inverse_shape[0]) * 4
        return int(flops), int(nbytes)

    def make_call(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        inverse_indices: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> PostPermuteCall:
        """Validate tensor-only state and construct the selection record."""
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
        if topk_weights.ndim != 2:
            raise ValueError("topk_weights must have shape [tokens, top_k]")
        if inverse_indices.ndim != 1 or inverse_indices.numel() != topk_weights.numel():
            raise ValueError("inverse_indices must have one entry per routing weight")
        if topk_weights.dtype is not torch.float32:
            raise TypeError("topk_weights must have dtype torch.float32")
        if inverse_indices.dtype is not torch.int32:
            raise TypeError("inverse_indices must have dtype torch.int32")
        if expert_output.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError("the current staged post-permute contract accepts BF16 or FP16 only")
        output_dtype = self.epilogue.resolve_output_dtype(expert_output.dtype)
        output_shape = (topk_weights.shape[0], expert_output.shape[-1])
        if out is not None and not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out is not None and (tuple(out.shape) != output_shape or out.dtype != output_dtype):
            raise ValueError("out shape and dtype must match the routing epilogue output")
        return PostPermuteCall(
            arch=get_sm_version(device.index),
            layout_key=self.layout.selection_key,
            max_m=self.layout.max_m,
            epilogue=self.epilogue,
            device_type=expert_output.device.type,
            input_dtype=expert_output.dtype,
            routing_weight_dtype=topk_weights.dtype,
            output_dtype=output_dtype,
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
        call = self.make_call(expert_output, topk_weights, inverse_indices, out)
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
    op = get_instance(instance_key)
    dtype = op.epilogue.resolve_output_dtype(expert_output.dtype)
    return torch.empty(
        (topk_weights.shape[0], expert_output.shape[-1]),
        dtype=dtype,
        device=expert_output.device,
    )


@torch.library.custom_op("tileops::moe_post_permute_fwd_inplace", mutates_args=("out",))
def _moe_post_permute_fwd_inplace(
    expert_output: torch.Tensor,
    inverse_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor,
    instance_key: str,
) -> None:
    get_instance(instance_key)._eager_forward(expert_output, inverse_indices, topk_weights, out=out)


@torch.library.custom_op("tileops::moe_grouped_gemm_fwd", mutates_args=())
def _moe_grouped_gemm_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(a, b, layout_metadata)


@_moe_grouped_gemm_fwd.register_fake
def _moe_grouped_gemm_fwd_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shape = op._infer_output_shapes(tuple(a.shape), tuple(b.shape), tuple(layout_metadata.shape))
    return torch.empty(shape["output"], dtype=op.resolve_output_dtype(a.dtype), device=a.device)


@torch.library.custom_op("tileops::moe_grouped_gemm_fwd_inplace", mutates_args=("out",))
def _moe_grouped_gemm_fwd_inplace(
    a: torch.Tensor,
    b: torch.Tensor,
    layout_metadata: torch.Tensor,
    out: torch.Tensor,
    instance_key: str,
) -> None:
    get_instance(instance_key)._eager_forward(a, b, layout_metadata, out=out)
