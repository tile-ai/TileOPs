"""Behavioral contract tests for the staged MoE public boundary."""

import dataclasses

import pytest
import torch

import tileops.ops.moe.staged as staged_module
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
from tileops.ops import moe as public_moe
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from tileops.ops.moe.contracts import (
    MaskedMetadata,
    PerRowExpertMetadata,
    PhysicalPsumMetadata,
    layout_from_preset,
    layout_value_guard,
)

pytestmark = pytest.mark.smoke

_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


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
    assert physical.metadata_length(rows=10, num_experts=3) == 3
    assert per_row.metadata_length(rows=10, num_experts=3) == 10
    assert masked.metadata_length(rows=12, num_experts=3) == 3
    with pytest.raises(ValueError, match="alignment > 1"):
        ContiguousLayoutSpec.aligned_per_row(1)
    with pytest.raises(ValueError, match="non-negative"):
        MaskedLayoutSpec(max_m=-1)

    with pytest.raises(ValueError, match="num_local_experts must be positive"):
        MoePrePermuteFwdOp(physical, num_local_experts=0)


def test_layout_presets_resolve_from_manifest_rows() -> None:
    """A workload row names a preset and carries only the argument that preset takes."""
    assert layout_from_preset("tight_physical_psum") == _TIGHT
    assert layout_from_preset(
        "aligned_per_row", alignment=128
    ) == ContiguousLayoutSpec.aligned_per_row(128)
    assert layout_from_preset("masked", max_m=64) == MaskedLayoutSpec(max_m=64)
    with pytest.raises(ValueError, match="neither alignment nor max_m"):
        layout_from_preset("tight_physical_psum", alignment=8)
    with pytest.raises(ValueError, match="takes alignment"):
        layout_from_preset("aligned_per_row")
    with pytest.raises(ValueError, match="takes max_m"):
        layout_from_preset("masked", alignment=8, max_m=4)
    with pytest.raises(ValueError, match="unknown layout preset"):
        layout_from_preset("padded")


def test_default_public_surface_hides_kernel_author_and_metadata_types() -> None:
    for name in (
        "MGroupedGemmCall",
        "PrePermuteCall",
        "PostPermuteCall",
        "PhysicalPsumMetadata",
        "PerRowExpertMetadata",
        "MaskedMetadata",
        "MaterializedExpertLayout",
        "NoScaleComputeSpec",
    ):
        assert not hasattr(public_moe, name)


def test_device_value_guards_cover_empty_experts_ordering_and_ranges() -> None:
    psum = PhysicalPsumMetadata(torch.tensor([0, 0, 3], dtype=torch.int32))
    assert psum.device_value_guard(materialized_rows=3).item()
    bad_psum = PhysicalPsumMetadata(torch.tensor([2, 1], dtype=torch.int32))
    assert not bad_psum.device_value_guard(materialized_rows=1).item()

    ids = PerRowExpertMetadata(torch.tensor([0, 0, 1, 1], dtype=torch.int32))
    assert ids.device_value_guard(num_experts=2).item()
    gap = PerRowExpertMetadata(torch.tensor([0, -1, 1], dtype=torch.int32))
    assert not gap.device_value_guard(num_experts=2).item()
    resumed = PerRowExpertMetadata(torch.tensor([0, 1, 0], dtype=torch.int32))
    assert not resumed.device_value_guard(num_experts=2).item()
    sentinel = PerRowExpertMetadata(torch.tensor([0, 1, 2, 2], dtype=torch.int32))
    assert not sentinel.device_value_guard(num_experts=2).item()
    assert sentinel.device_value_guard(num_experts=2, allow_capacity_sentinel=True).item()

    masked = MaskedMetadata(torch.tensor([0, 4, 5], dtype=torch.int32))
    assert not masked.device_value_guard(max_m=4).item()


@pytest.mark.parametrize(
    ("ends", "materialized_rows", "expected"),
    [
        pytest.param([], 0, True, id="zero-experts"),
        pytest.param([0, 0, 0], 0, True, id="all-empty"),
        pytest.param([0, 0, 3], 3, True, id="consecutive-empty"),
        pytest.param([0, 2, 1], 1, False, id="decreasing-end"),
        pytest.param([0, 2], 3, False, id="shape-not-authoritative-end"),
    ],
)
def test_physical_psum_guard_covers_empty_and_capacity_edges(
    ends: list[int], materialized_rows: int, expected: bool
) -> None:
    metadata = PhysicalPsumMetadata(torch.tensor(ends, dtype=torch.int32))
    assert metadata.device_value_guard(materialized_rows=materialized_rows).item() is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="guard test requires CUDA")
def test_device_value_validation_returns_a_device_guard_without_host_readback() -> None:
    metadata = PhysicalPsumMetadata(torch.tensor([0, 2], dtype=torch.int32, device="cuda"))
    guard = metadata.device_value_guard(materialized_rows=2)
    assert guard.device.type == "cuda"
    assert guard.dtype is torch.bool
    assert guard.shape == ()


def test_epilogue_spec_is_minimal_and_frozen_and_ops_type_check_their_layout() -> None:
    epilogue = RoutingEpilogueSpec()
    assert epilogue.accumulation_dtype is torch.float32
    assert epilogue.output_dtype is None
    assert epilogue.resolve_output_dtype(torch.bfloat16) is torch.bfloat16
    assert epilogue.resolve_output_dtype(torch.float16) is torch.float16
    assert RoutingEpilogueSpec(output_dtype=torch.float16).output_dtype is torch.float16
    with pytest.raises(ValueError, match="finite and positive"):
        RoutingEpilogueSpec(routed_scaling_factor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epilogue.routed_scaling_factor = 2.0
    with pytest.raises(TypeError, match="ContiguousLayoutSpec or MaskedLayoutSpec"):
        MoeGroupedGemmFwdOp(0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ContiguousLayoutSpec or MaskedLayoutSpec"):
        MoeExpertMLPFwdOp("tight_physical_psum")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out_dtype must be None"):
        MoeGroupedGemmFwdOp(_TIGHT, out_dtype=torch.float16)
    with pytest.raises(ValueError, match="activation must be None or one of"):
        MoeGroupedGemmFwdOp(_TIGHT, activation="relu")
    assert MoeGroupedGemmFwdOp(_TIGHT).resolve_output_dtype(torch.float16) is torch.float16
    assert (
        MoeGroupedGemmFwdOp(_TIGHT, out_dtype=torch.float32).resolve_output_dtype(torch.bfloat16)
        is torch.float32
    )
    with pytest.raises(TypeError, match="RoutingEpilogueSpec"):
        MoePostPermuteFwdOp(ContiguousLayoutSpec.tight_physical_psum(), epilogue=0)  # type: ignore[arg-type]


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
    # The op keys its cache on this record with ``m`` reset; every other field survives.
    taller = dataclasses.replace(gemm, m=4096)
    assert taller != gemm
    assert dataclasses.replace(taller, m=0) == dataclasses.replace(gemm, m=0)
    assert dataclasses.replace(gemm, n=8, m=0) != dataclasses.replace(gemm, m=0)
    assert dataclasses.replace(gemm, arch=100, m=0) != dataclasses.replace(gemm, m=0)
    # The flattened layout fields admit only what a layout spec can express.
    with pytest.raises(ValueError, match="no max_m"):
        dataclasses.replace(gemm, max_m=4)
    with pytest.raises(ValueError, match="alignment"):
        dataclasses.replace(gemm, packing="aligned")
    with pytest.raises(ValueError, match="carry no packing"):
        MGroupedGemmCall(arch=90, sm_count=1, kind="masked", packing="tight", max_m=4)
    with pytest.raises(ValueError, match="kind is"):
        MGroupedGemmCall(arch=90, sm_count=1, kind="padded")
    # Only the record that names no layout skips the checks.
    assert MGroupedGemmCall(arch=90, sm_count=1).kind == ""
    with pytest.raises(ValueError, match="kind is"):
        MGroupedGemmCall(arch=90, sm_count=1, max_m=4)


def _layout_key_of(call: MGroupedGemmCall) -> str | None:
    """The contiguous-layout key a test candidate claims, off the GEMM call."""
    if call.kind != "contiguous":
        return None
    return f"{call.packing}_{call.metadata_kind}"


class _PhysicalPsumCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        return _layout_key_of(call) == "tight_physical_psum"

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _PerRowCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        return _layout_key_of(call) == "tight_per_row"

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _GeneralCandidate(Kernel):
    general = True
    supported_archs = [90]

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _NeverCandidate(Kernel):
    @classmethod
    def applies(cls, call: object) -> bool:
        return False

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


def _zero_output(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None, dtype=None):
    """What an executable fake writes: zeros of the GEMM's output shape."""
    result = a.new_zeros((*a.shape[:-1], b.shape[1]), dtype=dtype)
    return result if out is None else out.copy_(result)


class _ExecutableGroupedCandidate(Kernel):
    builds = 0

    def __init__(self, call: MGroupedGemmCall) -> None:
        super().__init__()
        type(self).builds += 1
        self.call = call

    def forward(self, a, b, layout_metadata, *, out=None):
        return _zero_output(a, b, out, dtype=self.call.cd_dtype)


def _grouped_op_with_declared_candidates(**candidates: type[Kernel]) -> MoeGroupedGemmFwdOp:
    class DeclaringGroupedGemmOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return dict(candidates)

    return DeclaringGroupedGemmOp(_TIGHT)


def _tight_call(metadata_kind: str = "physical_psum", **fields: object) -> MGroupedGemmCall:
    return MGroupedGemmCall(
        arch=90,
        sm_count=1,
        kind="contiguous",
        packing="tight",
        metadata_kind=metadata_kind,
        **fields,
    )


def test_grouped_gemm_selection_behavior_table() -> None:
    physical_call = _tight_call()
    per_row_call = _tight_call("per_row")
    op = _grouped_op_with_declared_candidates(
        physical=_PhysicalPsumCandidate,
        per_row=_PerRowCandidate,
        general=_GeneralCandidate,
    )

    assert op.select_kernel_key(("physical", "per_row", "general"), physical_call) == "physical"
    assert op.select_kernel_key(("physical", "per_row", "general"), per_row_call) == "per_row"
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.select_kernel_key(
            ("physical", "per_row", "general"), dataclasses.replace(physical_call, arch=80)
        )


def test_grouped_gemm_ambiguous_and_incompatible_override_fail_explicitly() -> None:
    call = _tight_call()
    ambiguous = _grouped_op_with_declared_candidates(
        first=_PhysicalPsumCandidate,
        second=_PhysicalPsumCandidate,
    )
    with pytest.raises(ValueError, match="dispatch is ambiguous"):
        ambiguous.select_kernel_key(("first", "second"), call)

    class OverrideableOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {"special": _PhysicalPsumCandidate, "general": _GeneralCandidate}

    overridden = OverrideableOp(_TIGHT, kernel_map={"special": _NeverCandidate})
    with pytest.raises(ValueError, match="does not fall back"):
        overridden.select_kernel_key(("special", "general"), call)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_public_ops_build_complete_calls_before_selection() -> None:
    hidden = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32, device="cuda")
    pre_call = MoePrePermuteFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), num_local_experts=2
    ).make_call(hidden, topk_ids)
    assert (
        pre_call.num_experts,
        pre_call.num_tokens,
        pre_call.hidden_size,
        pre_call.top_k,
    ) == (2, 2, 8, 1)

    ends = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    a = torch.empty(2, 8, dtype=torch.bfloat16, device="cuda")
    b = torch.empty(2, 4, 8, dtype=torch.bfloat16, device="cuda")
    gemm_call = MoeGroupedGemmFwdOp(_TIGHT).make_call(a, b, ends)
    assert (gemm_call.m, gemm_call.num_groups, gemm_call.n, gemm_call.k) == (2, 2, 4, 8)
    assert (gemm_call.kind, gemm_call.packing, gemm_call.metadata_kind) == (
        "contiguous",
        "tight",
        "physical_psum",
    )
    assert (gemm_call.alignment, gemm_call.max_m) == (1, None)
    assert gemm_call.ab_dtype is gemm_call.cd_dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_call_architecture_comes_from_the_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    observed_indices: list[int | None] = []

    def fake_sm_version(index: int | None = None) -> int:
        observed_indices.append(index)
        return 90

    monkeypatch.setattr(staged_module, "get_sm_version", fake_sm_version)
    MoeGroupedGemmFwdOp(_TIGHT).make_call(
        torch.empty(1, 4, dtype=torch.bfloat16, device=device),
        torch.empty(1, 2, 4, dtype=torch.bfloat16, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
    )

    assert observed_indices == [device.index]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_staged_wiring_builds_all_family_calls_without_an_executable_candidate() -> None:
    device = torch.device("cuda")
    ends = torch.tensor([1, 2], dtype=torch.int32, device=device)
    expert_input = torch.empty(2, 8, dtype=torch.bfloat16, device=device)
    w_gate_up = torch.empty(2, 12, 8, dtype=torch.bfloat16, device=device)
    w_down = torch.empty(2, 8, 6, dtype=torch.bfloat16, device=device)
    mlp = MoeExpertMLPFwdOp(_TIGHT)

    gate_call = mlp.gate_up.make_call(expert_input, w_gate_up, ends)
    activated = torch.empty(2, 6, dtype=torch.bfloat16, device=device)
    down_call = mlp.down.make_call(activated, w_down, ends)

    assert gate_call.kind == down_call.kind == "contiguous"
    assert gate_call.packing == down_call.packing == "tight"
    assert (gate_call.k, gate_call.n, down_call.k, down_call.n) == (8, 12, 6, 8)
    # The gate_up GEMM carries the fused activation and hands the down GEMM ffn columns.
    assert (gate_call.activation, down_call.activation) == ("silu_and_mul", None)
    assert mlp.gate_up._infer_output_shapes((2, 8), (2, 12, 8), (2,)) == {"output": (2, 6)}
    assert tuple(mlp.kernel_delegates()) == (mlp.gate_up, mlp.down)
    with pytest.raises(ValueError, match="gated width"):
        mlp(
            expert_input, w_gate_up, torch.empty(2, 8, 5, dtype=torch.bfloat16, device=device), ends
        )

    inverse_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    post_call = MoePostPermuteFwdOp(_TIGHT, RoutingEpilogueSpec()).make_call(
        torch.empty(2, 8, dtype=torch.bfloat16, device=device),
        torch.ones(2, 1, dtype=torch.float32, device=device),
        inverse_indices,
    )
    assert post_call.layout_key == "tight_physical_psum"
    assert post_call.materialized_rows == 2
    assert (post_call.num_tokens, post_call.top_k, post_call.hidden_size) == (2, 1, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_post_permute_rejects_wrong_masked_geometry_with_same_row_count() -> None:
    device = torch.device("cuda")
    layout = MaskedLayoutSpec(max_m=4)
    inverse_indices = torch.tensor([0], dtype=torch.int32, device=device)
    wrong_geometry = torch.empty(1, 8, 4, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="masked expert_output"):
        MoePostPermuteFwdOp(layout).make_call(
            wrong_geometry,
            torch.ones(1, 1, dtype=torch.float32, device=device),
            inverse_indices,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
def test_injected_candidate_uses_common_selection_and_call_spec_cache() -> None:
    device = torch.device("cuda")
    ends = torch.tensor([1], dtype=torch.int32, device=device)
    a = torch.ones(1, 4, dtype=torch.bfloat16, device=device)
    b = torch.ones(1, 2, 4, dtype=torch.bfloat16, device=device)
    _ExecutableGroupedCandidate.builds = 0
    op = MoeGroupedGemmFwdOp(_TIGHT, kernel_map={"sm90_gemm": _ExecutableGroupedCandidate})

    first = op(a, b, ends)
    second = op(a, b, ends)
    # More rows for the same experts is the same specialization.
    taller = op(torch.ones(3, 4, dtype=torch.bfloat16, device=device), b, ends * 3)

    assert first.shape == second.shape == (1, 2)
    assert taller.shape == (3, 2)
    assert _ExecutableGroupedCandidate.builds == 1
    assert len(op.built_kernels("sm90_gemm")) == 1
    assert op.eval_roofline() == (2 * 3 * 2 * 4, (3 * 4 + 1 * 2 * 4 + 3 * 2) * 2 + 4)

    out = torch.empty(1, 2, dtype=torch.bfloat16, device=device)
    assert op(a, b, ends, out=out) is out
    with pytest.raises(ValueError, match="out must be"):
        op(a, b, ends, out=torch.empty(1, 2, dtype=torch.float32, device=device))


def test_expert_mlp_forwards_caller_replacements_to_both_gemms() -> None:
    mlp = MoeExpertMLPFwdOp(_TIGHT, kernel_map={"sm90_gemm": _ExecutableGroupedCandidate})
    assert mlp.gate_up.forwarded_overrides() == {"sm90_gemm": _ExecutableGroupedCandidate}
    assert mlp.down.forwarded_overrides() == {"sm90_gemm": _ExecutableGroupedCandidate}
    assert MoeExpertMLPFwdOp(_TIGHT, "gelu_and_mul").gate_up.activation == "gelu_and_mul"
    with pytest.raises(ValueError, match="activation must be one of"):
        MoeExpertMLPFwdOp(_TIGHT, "relu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
@pytest.mark.parametrize(
    "layout",
    [
        ContiguousLayoutSpec.tight_physical_psum(),
        ContiguousLayoutSpec.aligned_per_row(8),
    ],
)
def test_pre_permute_ships_one_contiguous_candidate(
    layout: ContiguousLayoutSpec,
) -> None:
    hidden = torch.empty(1, 4, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    op = MoePrePermuteFwdOp(layout, num_local_experts=1)
    call = op.make_call(hidden, topk_ids)
    assert op.select_kernel_key(tuple(op.kernel_map), call) == "contiguous"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_tight_pre_post_round_trip(dtype: torch.dtype) -> None:
    """The tensor-only staged boundary preserves every routed contribution."""
    tokens, top_k, experts, hidden = 4, 2, 4, 64
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    local_ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int32, device="cuda")
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    layout = ContiguousLayoutSpec.tight_physical_psum()

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device="cuda") // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])

    post = MoePostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
    assert pre.eval_roofline() == (0, 1616)

    assert post.eval_roofline() == (1024, 1600)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_aligned_per_row_pre_post_round_trip(dtype: torch.dtype) -> None:
    tokens, top_k, experts, hidden, alignment = 4, 2, 4, 64, 4
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    local_ids = torch.tensor([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=torch.int32, device="cuda")
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    layout = ContiguousLayoutSpec.aligned_per_row(alignment)
    capacity = tokens * top_k + experts * (alignment - 1)

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, row_expert_ids, inverse = pre(x, local_ids)
    assert expert_input.shape == (capacity, hidden)
    assert row_expert_ids.shape == (capacity,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device="cuda") // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])
    assert row_expert_ids.tolist() == [0] * 4 + [2] * 4 + [experts] * (capacity - 8)
    torch.testing.assert_close(expert_input[8:], torch.zeros_like(expert_input[8:]))

    post = MoePostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="make_call records CUDA architecture")
def test_grouped_gemm_make_call_checks_geometry_against_the_layout() -> None:
    """Shapes are validated against the declared layout, never used to guess it."""
    device = torch.device("cuda")
    b = torch.empty(2, 4, 8, dtype=torch.bfloat16, device=device)
    a = torch.empty(6, 8, dtype=torch.bfloat16, device=device)
    ends = torch.tensor([2, 6], dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match=r"layout_metadata must have shape \(2,\)"):
        MoeGroupedGemmFwdOp(_TIGHT).make_call(
            a, b, torch.zeros(6, dtype=torch.int32, device=device)
        )
    with pytest.raises(ValueError, match=r"layout_metadata must have shape \(6,\)"):
        MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_per_row()).make_call(a, b, ends)
    with pytest.raises(ValueError, match="multiple of the layout's alignment"):
        MoeGroupedGemmFwdOp(ContiguousLayoutSpec.aligned_per_row(4)).make_call(
            a, b, torch.zeros(6, dtype=torch.int32, device=device)
        )
    with pytest.raises(ValueError, match=r"masked a must have shape \[2, 4, k\]"):
        MoeGroupedGemmFwdOp(MaskedLayoutSpec(max_m=4)).make_call(a, b, ends)
    with pytest.raises(ValueError, match="contiguous a must have shape"):
        MoeGroupedGemmFwdOp(_TIGHT).make_call(a.reshape(2, 3, 8), b, ends)
    with pytest.raises(ValueError, match="same reduction dimension"):
        MoeGroupedGemmFwdOp(_TIGHT).make_call(
            torch.empty(6, 4, dtype=torch.bfloat16, device=device), b, ends
        )
    with pytest.raises(TypeError, match="BF16 or FP16"):
        MoeGroupedGemmFwdOp(_TIGHT).make_call(a.float(), b.float(), ends)
    with pytest.raises(TypeError, match="layout_metadata must have dtype torch.int32"):
        MoeGroupedGemmFwdOp(_TIGHT).make_call(a, b, ends.long())

    masked = MoeGroupedGemmFwdOp(MaskedLayoutSpec(max_m=3), out_dtype=torch.float32).make_call(
        a.reshape(2, 3, 8), b, ends
    )
    assert (masked.kind, masked.packing, masked.metadata_kind, masked.max_m) == (
        "masked",
        None,
        None,
        3,
    )
    assert (masked.m, masked.cd_dtype) == (6, torch.float32)

    aligned = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.aligned_per_row(2)).make_call(
        a, b, torch.zeros(6, dtype=torch.int32, device=device)
    )
    assert (aligned.packing, aligned.metadata_kind, aligned.alignment) == ("aligned", "per_row", 2)

    # A fused activation keeps b's stacked width in the call and halves the output.
    fused_op = MoeGroupedGemmFwdOp(_TIGHT, activation="silu_and_mul")
    fused = fused_op.make_call(a, b, ends)
    assert (fused.activation, fused.n) == ("silu_and_mul", 4)
    assert fused_op._infer_output_shapes((6, 8), (2, 4, 8), (2,)) == {"output": (6, 2)}
    with pytest.raises(ValueError, match="N must be even"):
        fused_op.make_call(a, torch.empty(2, 5, 8, dtype=torch.bfloat16, device=device), ends)
    with pytest.raises(ValueError, match=r"out must be a \[6, 2\] tensor"):
        fused_op.make_call(a, b, ends, out=torch.empty(6, 4, dtype=torch.bfloat16, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="selection records CUDA architecture")
def test_grouped_gemm_call_no_candidate_serves_reports_no_implementation() -> None:
    """A call outside every shipped candidate's region says so, rather than crashing."""
    device = torch.device("cuda")
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_per_row())  # not claimed yet
    assert set(op.kernel_map) == {"sm90_gemm"}
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(
            torch.empty(2, 8, dtype=torch.bfloat16, device=device),
            torch.empty(2, 4, 8, dtype=torch.bfloat16, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
        )


def _ids(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32)


@pytest.mark.parametrize(
    ("layout", "metadata", "rows", "expected"),
    [
        pytest.param(_TIGHT, [2, 2, 5], 5, True, id="tight-psum-ok"),
        pytest.param(_TIGHT, [2, 2, 4], 5, False, id="tight-psum-short"),
        pytest.param(
            ContiguousLayoutSpec.aligned_physical_psum(4),
            [2, 6, 12],
            12,
            True,
            id="aligned-psum-ok",
        ),
        pytest.param(
            ContiguousLayoutSpec.aligned_physical_psum(4),
            [2, 3, 12],
            12,
            False,
            id="aligned-psum-start-inside-tile",
        ),
        pytest.param(
            ContiguousLayoutSpec.aligned_physical_psum(4),
            [2, 6, 13],
            12,
            False,
            id="aligned-psum-past-capacity",
        ),
        pytest.param(
            ContiguousLayoutSpec.aligned_per_row(2),
            [0, 0, 1, 1, 3, 3],
            6,
            True,
            id="aligned-per-row-ok",
        ),
        pytest.param(
            ContiguousLayoutSpec.aligned_per_row(2),
            [0, 1, 1, 1, 3, 3],
            6,
            False,
            id="aligned-per-row-change-mid-tile",
        ),
        pytest.param(
            ContiguousLayoutSpec.tight_per_row(), [0, 0, 1, 2, 2], 5, True, id="tight-per-row-ok"
        ),
        pytest.param(
            ContiguousLayoutSpec.tight_per_row(),
            [0, 0, 1, 3, 3],
            5,
            False,
            id="tight-per-row-sentinel-not-allowed",
        ),
        pytest.param(MaskedLayoutSpec(max_m=4), [4, 0, 2], 12, True, id="masked-ok"),
        pytest.param(MaskedLayoutSpec(max_m=4), [5, 0, 2], 12, False, id="masked-over-capacity"),
    ],
)
def test_layout_value_guard_table(layout, metadata: list[int], rows: int, expected: bool) -> None:
    """Every layout's device-side invariants, answered without a host readback."""
    guard = layout_value_guard(layout, _ids(metadata), rows=rows, num_experts=3)
    assert guard.dtype is torch.bool and guard.shape == ()
    assert guard.item() is expected


def test_grouped_gemm_layout_guard_reads_rows_and_experts_off_the_operands() -> None:
    """The op's guard needs no call and no device: rows and experts come off ``a`` and ``b``."""
    op = MoeGroupedGemmFwdOp(MaskedLayoutSpec(max_m=4))
    a = torch.empty(3, 4, 8, dtype=torch.bfloat16)
    b = torch.empty(3, 2, 8, dtype=torch.bfloat16)
    assert op.layout_guard(a, b, _ids([4, 0, 2])).item()
    assert not op.layout_guard(a, b, _ids([4, 0, 5])).item()
    with pytest.raises(ValueError, match="length 3"):
        op.layout_guard(a, b, _ids([4, 0]))
