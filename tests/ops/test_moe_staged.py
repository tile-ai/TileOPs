"""Tests for the staged MoE public boundary and its grouped GEMM and expert MLP ops."""

import dataclasses

import pytest
import torch

from tileops.backend import BUILTIN, register_implementation, registry
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MGroupedGemmCall,
    MGroupedGemmFwdInterface,
    PostPermuteCall,
    PrePermuteCall,
)
from tileops.ops import moe as public_moe
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoEExpertMLPFwdOp,
    MoEGroupedGemmFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from tileops.utils import get_sm_version
from workloads.device import run_device, run_device_available
from workloads.moe import MoEExpertMLPWorkload, MoEGroupedGemmWorkload, post_permute_verification
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


@pytest.mark.smoke
def test_layout_presets_expose_only_supported_semantics() -> None:
    physical = ContiguousLayoutSpec.tight_physical_psum()
    per_row = ContiguousLayoutSpec.tight_per_row()
    aligned = ContiguousLayoutSpec.aligned_per_row(128)
    masked = MaskedLayoutSpec(max_m=4)

    assert physical.selection_key == "tight_physical_psum"
    assert per_row.selection_key == "tight_per_row"
    assert aligned.selection_key == "aligned_per_row"
    assert aligned.alignment == 128
    assert masked.selection_key == "masked_predicated"
    assert repr(physical) == "ContiguousLayoutSpec.tight_physical_psum()"
    assert repr(per_row) == "ContiguousLayoutSpec.tight_per_row()"
    assert repr(aligned) == "ContiguousLayoutSpec.aligned_per_row(128)"
    aligned_psum = ContiguousLayoutSpec.aligned_physical_psum(64)
    assert aligned_psum.selection_key == "aligned_physical_psum"
    assert repr(aligned_psum) == "ContiguousLayoutSpec.aligned_physical_psum(64)"
    assert (physical.kind, masked.kind) == ("contiguous", "masked")
    with pytest.raises(ValueError, match="alignment > 1"):
        ContiguousLayoutSpec.aligned_per_row(1)
    with pytest.raises(ValueError, match="non-negative"):
        MaskedLayoutSpec(max_m=-1)


@pytest.mark.smoke
def test_aligned_pre_permute_capacity_is_a_gemm_legal_row_extent() -> None:
    """Worst-case padding capacity still satisfies the aligned GEMM contract."""
    alignment = 128
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.aligned_per_row(alignment), num_local_experts=16)
    shapes = op._infer_output_shapes(
        (256, 2048), (256, 2), dtypes={"hidden_states": torch.bfloat16}
    )
    rows = shapes["expert_input"][0]
    assert rows == 2560
    assert rows % alignment == 0
    assert shapes["layout_metadata"] == (rows,)


@pytest.mark.smoke
def test_default_public_surface_hides_kernel_author_and_metadata_types() -> None:
    for name in (
        "MGroupedGemmCall",
        "PrePermuteCall",
        "PostPermuteCall",
        "MaterializedExpertLayout",
        "NoScaleComputeSpec",
    ):
        assert not hasattr(public_moe, name)


@pytest.mark.smoke
def test_epilogue_spec_is_minimal_and_frozen() -> None:
    epilogue = RoutingEpilogueSpec()
    assert epilogue.accumulation_dtype is torch.float32
    # The output dtype is the op's ``out_dtype``, not the epilogue's.
    assert not hasattr(epilogue, "output_dtype")
    with pytest.raises(ValueError, match="finite and positive"):
        RoutingEpilogueSpec(routed_scaling_factor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epilogue.routed_scaling_factor = 2.0


@pytest.mark.smoke
def test_family_call_specs_are_frozen_and_keep_selection_axes_separate() -> None:
    pre = PrePermuteCall(arch=90, sm_count=1, layout=ContiguousLayoutSpec.tight_physical_psum())
    aligned_pre = dataclasses.replace(pre, layout=ContiguousLayoutSpec.aligned_per_row(128))
    gemm = MGroupedGemmCall(
        arch=90, sm_count=1, kind="contiguous", packing="tight", metadata_kind="physical_psum"
    )
    post = PostPermuteCall(
        arch=90,
        sm_count=1,
        layout_key="tight_physical_psum",
        epilogue=RoutingEpilogueSpec(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pre.arch = 100
    assert aligned_pre != pre
    assert len({pre, aligned_pre}) == 2
    assert post.layout_key == "tight_physical_psum"
    # ``m`` selects but does not build, and the device facts follow from ``device``, so
    # they are outside this record's identity; every other field is inside it.
    taller = dataclasses.replace(gemm, m=4096)
    assert taller == gemm
    assert len({gemm, taller}) == 1
    assert dataclasses.replace(gemm, n=8) != gemm
    assert dataclasses.replace(gemm, device=torch.device("meta")) != gemm
    assert dataclasses.replace(gemm, arch=100) == gemm


_CANDIDATE_KEY = "grouped_gemm_candidate"


class _ExecutableGroupedCandidate(Kernel, MGroupedGemmFwdInterface):
    """A registered implementation that writes zeros of the GEMM's output shape instead of
    compiling, preferred over every in-tree one."""

    builds = 0
    preferred_over = frozenset(MoEGroupedGemmFwdOp.kernel_types)

    def __init__(self, call: MGroupedGemmCall) -> None:
        super().__init__()
        type(self).builds += 1
        self.call = call

    def forward(self, a, b, layout_metadata, out=None):
        zeros = a.new_zeros((*a.shape[:-1], b.shape[1]), dtype=self.call.cd_dtype)
        return zeros if out is None else out.copy_(zeros)


@pytest.fixture
def registered_candidate():
    """``_ExecutableGroupedCandidate`` registered for every grouped GEMM this test builds."""
    state = registry.snapshot()
    register_implementation("MoEGroupedGemmFwdOp", _CANDIDATE_KEY, _ExecutableGroupedCandidate)
    yield
    registry.restore(state)


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="candidate test uses CUDA calls")
def test_injected_candidate_uses_common_selection_and_call_spec_cache(registered_candidate) -> None:
    device = torch.device("cuda")
    ends = torch.tensor([1], dtype=torch.int32, device=device)
    # A shape the in-tree kernel serves too: the registered candidate is preferred over it.
    a = torch.ones(1, 8, dtype=torch.bfloat16, device=device)
    b = torch.ones(1, 8, 8, dtype=torch.bfloat16, device=device)
    _ExecutableGroupedCandidate.builds = 0
    op = MoEGroupedGemmFwdOp(_TIGHT, target=BUILTIN)
    op.autotune()

    first = op(a, b, ends)
    second = op(a, b, ends)
    # More rows for the same experts is the same specialization.
    taller = op(torch.ones(3, 8, dtype=torch.bfloat16, device=device), b, ends * 3)

    assert first.shape == second.shape == (1, 8)
    assert taller.shape == (3, 8)
    assert op.eval_roofline() == (2 * 3 * 8 * 8, (3 * 8 + 1 * 8 * 8 + 3 * 8) * 2 + 4)

    out = torch.empty(1, 8, dtype=torch.bfloat16, device=device)
    assert op(a, b, ends, out=out) is out
    with pytest.raises(ValueError, match="out does not have the dtype of output"):
        op(a, b, ends, out=torch.empty(1, 8, dtype=torch.float32, device=device))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_expert_mlp_installs_a_registered_implementation_in_both_gemms(
    registered_candidate,
) -> None:
    mlp = MoEExpertMLPFwdOp(_TIGHT)
    assert mlp.gate_up._registered[_CANDIDATE_KEY] is _ExecutableGroupedCandidate
    assert mlp.down._registered[_CANDIDATE_KEY] is _ExecutableGroupedCandidate
    assert MoEExpertMLPFwdOp(_TIGHT, "gelu_and_mul").gate_up.activation == "gelu_and_mul"


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_tight_pre_post_round_trip(dtype: torch.dtype) -> None:
    """The tensor-only staged boundary preserves every routed contribution."""
    tokens, top_k, experts, hidden = 4, 2, 4, 64
    x = torch.randn(tokens, hidden, dtype=dtype, device=run_device())
    local_ids = torch.tensor(
        [[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int32, device=run_device()
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.tight_physical_psum()

    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])

    post = MoEPostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())
    assert pre.eval_roofline() == (0, 1616)
    assert post.eval_roofline() == (1024, 1600)

    # A routing scale other than one multiplies every output element once more.
    scaled = MoEPostPermuteFwdOp(layout, RoutingEpilogueSpec(routed_scaling_factor=2.0))
    output = scaled(expert_input, weights, inverse)
    compare_outputs(
        output.float(),
        2 * expected,
        post_permute_verification(scaled.epilogue.routed_scaling_factor),
    )
    assert scaled.eval_roofline() == (1024 + tokens * hidden, 1600)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden",
    [(512, 8, 128, 128), (32, 8, 128, 7168)],
)
def test_staged_tight_optimized_shapes_round_trip(
    tokens: int, top_k: int, experts: int, hidden: int
) -> None:
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=run_device())
    local_ids = (
        torch.arange(tokens * top_k, dtype=torch.int32, device=run_device())
        .remainder(experts)
        .reshape(tokens, top_k)
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)

    expert_input, physical_ends, inverse = pre(x, local_ids)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])
    counts = torch.bincount(local_ids.flatten().long(), minlength=experts)
    assert torch.equal(physical_ends, counts.cumsum(0).int())
    output = MoEPostPermuteFwdOp(layout)(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_aligned_per_row_pre_post_round_trip(dtype: torch.dtype) -> None:
    tokens, top_k, experts, hidden, alignment = 4, 2, 4, 64, 4
    x = torch.randn(tokens, hidden, dtype=dtype, device=run_device())
    local_ids = torch.tensor(
        [[0, 0], [0, 2], [2, 2], [2, 0]], dtype=torch.int32, device=run_device()
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.aligned_per_row(alignment)
    capacity = tokens * top_k + experts * (alignment - 1)

    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, row_expert_ids, inverse = pre(x, local_ids)
    assert expert_input.shape == (capacity, hidden)
    assert row_expert_ids.shape == (capacity,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])
    assert row_expert_ids.tolist() == [0] * 4 + [2] * 4 + [experts] * (capacity - 8)
    assert torch.count_nonzero(expert_input[8:]) == 0

    post = MoEPostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())


# MoEGroupedGemmFwdOp selects MoEGroupedGemmKernel, the adapter over the shared template, through the
# staged candidate protocol; these tests route every layout it claims through the op and check the
# rows the layout defines against the workload's per-expert reference.


_TIGHT_MOE_GROUPED_GEMM = {
    "contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}
}


def _aligned(metadata_kind: str) -> dict:
    return {"contiguous": {"packing": "aligned", "metadata_kind": metadata_kind, "alignment": 128}}


def _gemm_call(dtype: torch.dtype, layout: dict, **row):
    return manifest_call(
        "MoEGroupedGemmFwdOp", {"D": str(dtype).removeprefix("torch.")}, layout=layout, **row
    )


def _run(workload) -> tuple:
    """The op built from the call, its output on the call's inputs, and the reference."""
    inputs = workload.gen_inputs()
    cls = MoEGroupedGemmFwdOp if len(inputs) == 3 else MoEExpertMLPFwdOp
    op = cls(**workload.call.arguments({}))
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    return op, out


@pytest.mark.parametrize(
    "layout,rows,dtype",
    [
        pytest.param(
            _TIGHT_MOE_GROUPED_GEMM,
            {"P": 600},
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="tight-psum",
        ),
        pytest.param(
            _aligned("per_row"),
            {"P": 128 * 9},
            torch.float16,
            marks=pytest.mark.smoke,
            id="aligned-per-row",
        ),
        pytest.param(
            _aligned("physical_psum"),
            {"P": 128 * 9},
            torch.bfloat16,
            marks=pytest.mark.full,
            id="aligned-psum",
        ),
        pytest.param(
            {"masked": {"max_m": 128}}, {}, torch.bfloat16, marks=pytest.mark.full, id="masked"
        ),
    ],
)
def test_grouped_gemm_runs_each_layout_through_the_op(layout, rows, dtype):
    """Each claimed layout: the op builds the device's kernel once and matches the reference."""
    num_experts = 6
    op, out = _run(
        MoEGroupedGemmWorkload(_gemm_call(dtype, layout, K=512, E=num_experts, N=256, **rows))
    )
    assert out.dtype is dtype and out.shape[-1] == 256


def _uneven_rows(layout: dict, sizes: list) -> tuple:
    """Row count and metadata placing ``sizes[e]`` rows on expert ``e``, empty experts included.

    Tight rows pack; an aligned expert starts on a multiple of the alignment; per-row ids
    pad each expert to the alignment and close with one tile of unassigned rows; a masked
    slab records each expert's count.
    """
    if "masked" in layout:
        return len(sizes) * layout["masked"]["max_m"], list(sizes)
    spec = layout["contiguous"]
    align = spec["alignment"]
    if spec["metadata_kind"] == "per_row":
        ids = [g for g, size in enumerate(sizes) for _ in range(-(-size // align) * align)]
        ids += [len(sizes)] * align
        return len(ids), ids
    row, ends = 0, []
    for size in sizes:
        row = (row if spec["packing"] == "tight" else -(-row // align) * align) + size
        ends.append(row)
    return (row if spec["packing"] == "tight" else -(-row // align) * align), ends


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("layout", "sizes", "activation"),
    [
        pytest.param(_TIGHT_MOE_GROUPED_GEMM, [100, 0, 300, 128, 7, 64], None, id="tight-psum"),
        pytest.param(_aligned("per_row"), [100, 0, 300, 128, 7, 64], None, id="aligned-per-row"),
        pytest.param(_aligned("physical_psum"), [100, 0, 300, 128, 7, 64], None, id="aligned-psum"),
        pytest.param(_aligned("physical_psum"), [300] * 32, None, id="aligned-psum-32-experts"),
        pytest.param({"masked": {"max_m": 256}}, [100, 0, 256, 33], None, id="masked"),
        pytest.param(
            {"masked": {"max_m": 256}}, [100, 0, 256, 33], "silu_and_mul", id="masked-gated"
        ),
    ],
)
def test_grouped_gemm_writes_every_defined_row_of_uneven_experts(layout, sizes, activation):
    """Uneven experts, an empty one included, into a poisoned ``out``: every defined row is
    written and matches the reference."""
    rows, metadata = _uneven_rows(layout, sizes)
    extra = {} if activation is None else {"activation": activation}
    rows_arg = {} if "masked" in layout else {"P": rows}
    call = _gemm_call(torch.bfloat16, layout, K=512, E=len(sizes), N=256, **rows_arg, **extra)
    workload = MoEGroupedGemmWorkload(call)
    a, b, _ = workload.gen_inputs()
    metadata = torch.tensor(metadata, dtype=torch.int32, device=a.device)
    out = torch.full((*a.shape[:-1], 256), 1e4, dtype=torch.bfloat16, device=a.device)
    op = MoEGroupedGemmFwdOp(**call.arguments({}))
    assert op(a, b, metadata, out=out) is out
    compare_outputs(
        out, workload.ref_program(a, b, metadata), workload.verification(a, b, metadata)
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_grouped_gemm_tight_per_row_matches_the_psum_layout():
    """Per-row ids on tight rows, an empty group included: the psum layout's exact output."""
    if get_sm_version(torch.device(run_device()).index) != 90:
        pytest.skip("only the SM90 template serves a tight per-row layout")
    sizes = [100, 0, 300, 128, 7, 64]
    tight_per_row = {"contiguous": {"packing": "tight", "metadata_kind": "per_row", "alignment": 1}}
    call = _gemm_call(torch.bfloat16, tight_per_row, K=1024, E=len(sizes), N=2048, P=sum(sizes))
    workload = MoEGroupedGemmWorkload(call)
    a, b, _ = workload.gen_inputs()
    counts = torch.tensor(sizes, device=a.device)
    ids = torch.repeat_interleave(torch.arange(len(sizes), device=a.device), counts).int()
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_per_row())
    out = op(a, b, ids)
    compare_outputs(out, workload.ref_program(a, b, ids), workload.verification(a, b, ids))
    psum = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    assert torch.equal(out, psum(a, b, counts.cumsum(0).int()))


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_grouped_gemm_fuses_the_gated_activation(activation):
    """With ``activation`` the op hands the kernel a gate||up ``b`` and gets ffn columns back."""
    num_experts = 6
    call = _gemm_call(
        torch.bfloat16,
        _TIGHT_MOE_GROUPED_GEMM,
        P=600,
        K=512,
        E=num_experts,
        N=192,
        activation=activation,
    )
    op, out = _run(MoEGroupedGemmWorkload(call))
    assert out.shape == (600, 192)


@pytest.mark.smoke
def test_grouped_gemm_dims_off_the_tile_grid():
    """Cover non-tile-aligned K and N."""
    num_experts = 6
    _run(
        MoEGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=200, K=96, E=num_experts, N=192)
        )
    )


@pytest.mark.smoke
def test_grouped_gemm_fp32_output_and_preallocated_out():
    num_experts = 6
    call = _gemm_call(
        torch.bfloat16,
        _TIGHT_MOE_GROUPED_GEMM,
        P=600,
        K=512,
        E=num_experts,
        N=256,
        out_dtype="float32",
    )
    workload = MoEGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoEGroupedGemmFwdOp(**call.arguments({}))
    out = torch.empty(600, 256, dtype=torch.float32, device=run_device())
    assert op(a, b, metadata, out=out) is out
    compare_outputs(
        out, workload.ref_program(a, b, metadata), workload.verification(a, b, metadata)
    )


@pytest.mark.smoke
def test_grouped_gemm_refuses_a_strided_out():
    """The output buffer is declared contiguous; a strided one is refused before the kernel."""
    workload = MoEGroupedGemmWorkload(
        _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=64, K=64, E=2, N=64)
    )
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    out = torch.empty(64, 128, dtype=torch.bfloat16, device=run_device())[:, ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        op(*workload.gen_inputs(), out=out)


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype,activation",
    [
        pytest.param(torch.bfloat16, "silu_and_mul", id="bf16-silu"),
        pytest.param(torch.float16, "gelu_and_mul", id="fp16-gelu"),
    ],
)
def test_expert_mlp_composes_two_template_gemms(dtype, activation):
    """The MLP is two template GEMMs; the first carries the activation and halves its width."""
    num_experts = 6
    call = manifest_call(
        "MoEExpertMLPFwdOp",
        {"D": str(dtype).removeprefix("torch.")},
        layout=_TIGHT_MOE_GROUPED_GEMM,
        activation=activation,
        P=600,
        H=256,
        E=num_experts,
        F=192,
    )
    op, out = _run(MoEExpertMLPWorkload(call))
    assert out.dtype is dtype and out.shape == (600, 256)


def _grouped_call(**facts) -> MGroupedGemmCall:
    return MGroupedGemmCall(
        **{
            "arch": 90,
            "sm_count": 132,
            "kind": "contiguous",
            "packing": "tight",
            "metadata_kind": "physical_psum",
            "ab_dtype": torch.bfloat16,
            "cd_dtype": torch.bfloat16,
            "num_groups": 2,
            "m": 8,
            "n": 16,
            "k": 64,
            **facts,
        }
    )


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("layout", "activation", "facts", "reason"),
    [
        pytest.param("tight", None, {"k": 60}, "steps K by 8", id="k-off-8"),
        pytest.param(
            "tight",
            "silu_and_mul",
            {"n": 24, "activation": "silu_and_mul"},
            "N a multiple of 16",
            id="fused-n-off-16",
        ),
        pytest.param(
            "aligned8",
            None,
            {"packing": "aligned", "metadata_kind": "per_row", "alignment": 8},
            "tile height",
            id="alignment-off-a-tile",
        ),
    ],
)
def test_grouped_gemm_refuses_what_the_template_cannot_run(layout, activation, facts, reason):
    spec = (
        ContiguousLayoutSpec.aligned_per_row(8)
        if layout == "aligned8"
        else ContiguousLayoutSpec.tight_physical_psum()
    )
    op = MoEGroupedGemmFwdOp(spec, **({"activation": activation} if activation else {}))
    with pytest.raises(ValueError, match=reason):
        op.key_for("grouped_gemm", _grouped_call(**facts))
