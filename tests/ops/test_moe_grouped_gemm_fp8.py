"""Tests for the block-scaled FP8 M-grouped GEMM op."""

import itertools
import math

import pytest
import torch

from tileops._kernels.moe import (
    MGroupedGEMMFP8Call,
    MoEGroupedGEMMFP8Kernel,
    MoEGroupedGEMMFP8MMAKernel,
)
from tileops._kernels.moe.moe_grouped_gemm_fp8 import _smem_bytes
from tileops._ops.moe import MaskedLayoutSpec, MoEGroupedGEMMFP8FwdOp
from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from tileops.manifest import load_workloads
from workloads.moe import MoEGroupedGEMMFP8Workload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

_OP = "MoEGroupedGEMMFP8FwdOp"
_KEY = "grouped_gemm_fp8"
# An output value no valid row produces.
_SENTINEL = -12288.0


def _workload(max_m: int, **dims) -> MoEGroupedGEMMFP8Workload:
    # T only feeds the generated counts; half the slabs' capacity keeps every count in range.
    dims.setdefault("T", dims["E"] * max_m // 2)
    return MoEGroupedGEMMFP8Workload(
        manifest_call(_OP, layout={"masked": {"max_m": max_m}}, **dims)
    )


def _with_counts(workload: MoEGroupedGEMMFP8Workload, counts: list) -> tuple:
    """The workload's inputs with *counts* valid rows per expert in place of the generated ones."""
    *operands, metadata = workload.gen_inputs()
    return (*operands, torch.tensor(counts, dtype=torch.int32, device=metadata.device))


def _run_into_guarded_out(op, workload: MoEGroupedGEMMFP8Workload, *inputs) -> torch.Tensor:
    """Check output reuse, guard integrity and correctness on valid rows."""
    a, b, metadata = inputs[0], inputs[2], inputs[-1]
    shape = (*a.shape[:-1], b.shape[1])
    guard = 4096
    flat = torch.full(
        (math.prod(shape) + guard,),
        _SENTINEL,
        dtype=torch.bfloat16,
        device=a.device,
    )
    out = flat[:-guard].view(shape)
    got = op(*inputs, out=out)
    assert got is out
    assert bool((flat[-guard:] == _SENTINEL).all()), "a store ran past the output"
    reference_inputs = (*inputs[:-1], metadata.clamp(0, a.shape[1])) if a.ndim == 3 else inputs
    compare_outputs(
        got, workload.ref_program(*reference_inputs), workload.verification(*reference_inputs)
    )
    return got


_ROWS = {row["label"]: row for row in load_workloads(_OP)}


@pytest.mark.parametrize(
    "label,arch",
    [
        pytest.param(label, arch, marks=[tier, getattr(pytest.mark, f"sm{arch}"), *extra])
        for label, tier, extra in [
            # Aligned per-row rows: the SM90 kernel and the SM89 MMA kernel serve them in-tree.
            ("qwen3-30b-ep8-skew-down", pytest.mark.smoke, [pytest.mark.in_tree_kernels]),
            ("ds-v3-e8-up-masked", pytest.mark.smoke, []),
            ("qwen3-235b-dec-b64-down", pytest.mark.full, []),
            ("kimi-k2-dec-b256-up", pytest.mark.full, []),
            ("kimi-k2-dec-b64-down", pytest.mark.full, []),
            ("ds-v3-ep8-prefill-down", pytest.mark.full, [pytest.mark.in_tree_kernels]),
        ]
        for arch in (89, 90)
    ],
)
def test_fp8_grouped_gemm_runs_the_manifest_rows(label: str, arch: int) -> None:
    """Representative manifest rows, with their generated valid counts, match the reference."""
    row = {key: value for key, value in _ROWS[label].items() if key != "label"}
    workload = MoEGroupedGEMMFP8Workload(manifest_call(_OP, **row))
    inputs = workload.gen_inputs()
    out = MoEGroupedGEMMFP8FwdOp(**workload.call.arguments({}))(*inputs)
    assert out.dtype is torch.bfloat16
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("max_m", "dims", "counts"),
    [
        pytest.param(64, {"N": 128, "K": 128}, [64, 0, 1, 99], id="max_m-64-n128-k128"),
        pytest.param(65, {"K": 256}, [65, 0, 64, 1], id="max_m-65"),
        pytest.param(128, {}, [128, -1000, 1, 0], id="max_m-128"),
        pytest.param(129, {}, [0, 1, 64, 65, 129, 200], id="max_m-129"),
        pytest.param(100, {}, [100, 0, 100], id="max_m-100"),
        pytest.param(1152, {"K": 128}, [1100], id="max_m-1152-e1"),
        pytest.param(0, {"N": 128, "K": 128}, [0, 0], id="max_m-0"),
    ],
)
def test_fp8_grouped_gemm_valid_rows_at_tile_and_slab_boundaries(
    max_m: int, dims: dict, counts: list, arch: int
) -> None:
    dims = {"N": 256, "K": 512, **dims}
    workload = _workload(max_m, E=len(counts), **dims)
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=max_m))
    _run_into_guarded_out(op, workload, *_with_counts(workload, counts))


def _forced_op(layout, config: dict) -> MoEGroupedGEMMFP8FwdOp:
    """Use the in-tree kernel with an explicit launch configuration."""

    class Forced(MoEGroupedGEMMFP8Kernel):
        preferred_over = frozenset(MoEGroupedGEMMFP8FwdOp.kernel_types)

        @property
        def default_config(self) -> dict:
            return dict(config)

    state = registry.snapshot()
    try:
        register_kernel_type(_OP, f"{_KEY}_forced", Forced)
        return MoEGroupedGEMMFP8FwdOp(layout)
    finally:
        registry.restore(state)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("block_n", "num_stages", "n", "stage_scales"),
    [
        pytest.param(160, 4, 512, 0, id="bn160-static-ring"),
        pytest.param(192, 3, 512, 0, id="bn192-index-ring"),
        pytest.param(160, 4, 512, 1, id="bn160-static-ring-staged-scales"),
        pytest.param(192, 3, 512, 1, id="bn192-index-ring-staged-scales"),
        pytest.param(192, 4, 384, 0, id="bn192-two-tiles-n384"),
        pytest.param(192, 4, 256, 0, id="bn192-two-tiles-n256"),
        pytest.param(160, 4, 256, 0, id="bn160-two-tiles-n256"),
    ],
)
def test_fp8_grouped_gemm_wide_tiles_clip_at_n(
    block_n: int, num_stages: int, n: int, stage_scales: int
) -> None:
    """N/M tails and empty experts preserve guarded output across persistent waves."""
    counts = [192, 0, 70]
    workload = _workload(192, E=len(counts), N=n, K=512)
    config = {
        "block_m": 128,
        "block_n": block_n,
        "num_stages": num_stages,
        "num_sms": 3,
        "stage_scales": stage_scales,
    }
    op = _forced_op(MaskedLayoutSpec(max_m=192), config)
    out = _run_into_guarded_out(op, workload, *_with_counts(workload, counts))
    assert bool((out[1] == _SENTINEL).all()) and bool((out[2, 128:] == _SENTINEL).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_staged_scales_on_one_warp_group() -> None:
    """Staged scales remain correct across ring reuse and partial tiles."""
    counts = [128, 0, 70]
    workload = _workload(128, E=len(counts), N=256, K=1792)
    config = {"block_m": 64, "block_n": 128, "num_stages": 7, "num_sms": 3, "stage_scales": 1}
    op = _forced_op(MaskedLayoutSpec(max_m=128), config)
    out = _run_into_guarded_out(op, workload, *_with_counts(workload, counts))
    assert bool((out[1] == _SENTINEL).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_tma_a_scales_on_a_deep_ring() -> None:
    """TMA-loaded A scales stay correct as their group buffers rotate across a seven-stage ring."""
    counts = [128, 0, 70]
    workload = _workload(128, E=len(counts), N=256, K=2048)
    config = {"block_m": 64, "block_n": 128, "num_stages": 7, "num_sms": 3, "stage_scales": 2}
    op = _forced_op(MaskedLayoutSpec(max_m=128), config)
    out = _run_into_guarded_out(op, workload, *_with_counts(workload, counts))
    assert bool((out[1] == _SENTINEL).all())


def _layout(packing: str, metadata_kind: str, alignment: int = 1) -> dict:
    return {
        "contiguous": {"packing": packing, "metadata_kind": metadata_kind, "alignment": alignment}
    }


def _contiguous_workload(layout: dict, sizes: list, **dims) -> MoEGroupedGEMMFP8Workload:
    """Build contiguous expert rows with explicit per-expert sizes."""
    spec = layout["contiguous"]
    a = spec["alignment"]
    rows = sum(sizes) if spec["packing"] == "tight" else sum(-(-n // a) * a for n in sizes)
    stated = {"P": rows} if spec["metadata_kind"] == "physical_psum" else {}
    dims = {"N": 256, "K": 512, **dims}
    return MoEGroupedGEMMFP8Workload(
        manifest_call(_OP, layout=layout, E=len(sizes), sizes=sizes, **stated, **dims)
    )


def _scaled_inputs(workload: MoEGroupedGEMMFP8Workload) -> tuple:
    """Different K and N scale blocks expose swapped or missing scale indices."""
    a, sa, b, sb, ids = workload.gen_inputs()
    sa.copy_(torch.arange(sa.numel(), device=sa.device).reshape(sa.shape).remainder(7) / 8 + 0.5)
    sb.copy_(torch.arange(sb.numel(), device=sb.device).reshape(sb.shape).remainder(11) / 8 + 0.5)
    return a, sa, b, sb, ids


# Partial last tiles, an empty expert, a one-row expert, and runs over several tiles.
_SIZES = [5, 0, 77, 1, 300, 64, 129]


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("layout", "sizes", "config"),
    [
        pytest.param(
            _layout("tight", "physical_psum"),
            _SIZES,
            None,
            marks=pytest.mark.smoke,
            id="tight-psum",
        ),
        pytest.param(
            _layout("tight", "per_row"),
            _SIZES,
            {"block_m": 64, "block_n": 160, "num_stages": 4, "num_sms": 5, "stage_scales": 1},
            marks=pytest.mark.full,
            id="tight-per-row-bm64-bn160-staged",
        ),
        pytest.param(
            _layout("tight", "per_row"),
            _SIZES,
            {"block_m": 128, "block_n": 192, "num_stages": 4, "num_sms": 5, "stage_scales": 2},
            marks=pytest.mark.smoke,
            id="tight-per-row-tma-scales",
        ),
        pytest.param(
            _layout("aligned", "per_row", 128),
            _SIZES,
            {"block_m": 128, "block_n": 192, "num_stages": 4, "num_sms": 5, "stage_scales": 2},
            marks=pytest.mark.smoke,
            id="aligned-per-row-tma-scales",
        ),
        pytest.param(
            _layout("aligned", "physical_psum", 128),
            _SIZES,
            None,
            marks=pytest.mark.full,
            id="aligned-psum-128",
        ),
        pytest.param(
            _layout("aligned", "per_row", 128),
            _SIZES,
            None,
            marks=pytest.mark.full,
            id="aligned-per-row-128",
        ),
        pytest.param(
            _layout("aligned", "physical_psum", 64),
            _SIZES,
            {"block_m": 64, "block_n": 128, "num_stages": 4, "num_sms": 7, "stage_scales": 0},
            marks=pytest.mark.full,
            id="aligned-psum-64-bm64",
        ),
        pytest.param(
            _layout("tight", "per_row"),
            [2200, 3],
            {"block_m": 128, "block_n": 192, "num_stages": 4, "num_sms": 9, "stage_scales": 0},
            marks=pytest.mark.full,
            id="tight-per-row-18-tiles",
        ),
    ],
)
def test_fp8_grouped_gemm_contiguous_rows_of_every_expert(
    layout: dict, sizes: list, config: "dict | None"
) -> None:
    workload = _contiguous_workload(layout, sizes)
    spec = workload.call.arguments({})["layout"]
    op = MoEGroupedGEMMFP8FwdOp(spec) if config is None else _forced_op(spec, config)
    _run_into_guarded_out(op, workload, *workload.gen_inputs())


@pytest.mark.sm89
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "alignment,block_m,tune",
    [
        pytest.param(32, 32, False, marks=pytest.mark.smoke, id="align32-bm32"),
        pytest.param(64, 64, False, marks=pytest.mark.full, id="align64-bm64"),
        pytest.param(128, 32, False, marks=pytest.mark.full, id="align128-bm32"),
        pytest.param(32, 32, True, marks=pytest.mark.full, id="align32-tuned"),
    ],
)
def test_fp8_grouped_gemm_mma_aligned_configs(alignment: int, block_m: int, tune: bool) -> None:
    """Default, explicit and tuned tile heights respect the segment alignment."""
    workload = _contiguous_workload(_layout("aligned", "per_row", alignment), _SIZES, N=256, K=256)
    op = MoEGroupedGEMMFP8FwdOp(workload.call.arguments({})["layout"], target=BUILTIN)
    if tune:
        op.request_tune()
    inputs = _scaled_inputs(workload)
    _run_into_guarded_out(op, workload, *inputs)
    kernel = next(op.iter_kernels())
    assert type(kernel) is MoEGroupedGEMMFP8MMAKernel
    assert all(alignment % cfg["block_m"] == 0 for cfg in kernel.autotune_configs)
    if tune:
        assert kernel.config["block_m"] == block_m
    else:
        kernel.init_config({"block_m": block_m, "block_n": 128})
        _run_into_guarded_out(op, workload, *inputs)


@pytest.mark.sm89
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_mma_dynamic_rows() -> None:
    """One compiled kernel serves changed row counts, including an empty call."""
    op = None
    identity = None
    for sizes in ([5, 0, 77], [257, 1, 64], [0, 0, 0], [1, 0, 33]):
        workload = _contiguous_workload(_layout("aligned", "per_row", 128), sizes, N=256, K=256)
        if op is None:
            op = MoEGroupedGEMMFP8FwdOp(workload.call.arguments({})["layout"], target=BUILTIN)
        _run_into_guarded_out(op, workload, *_scaled_inputs(workload))
        kernel = next(op.iter_kernels())
        if identity is None:
            identity = kernel
        assert kernel is identity


@pytest.mark.sm89
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_mma_rejects_unaligned_tiles() -> None:
    """An explicit tile cannot straddle two aligned expert segments."""
    workload = _contiguous_workload(_layout("aligned", "per_row", 32), _SIZES, N=256, K=256)
    op = MoEGroupedGEMMFP8FwdOp(workload.call.arguments({})["layout"], target=BUILTIN)
    _run_into_guarded_out(op, workload, *_scaled_inputs(workload))
    with pytest.raises(ValueError, match="must divide"):
        next(op.iter_kernels()).init_config({"block_m": 64, "block_n": 64})


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_skips_aligned_padding_past_every_run(arch: int) -> None:
    """Rows past every run carry the id ``E``: their tiles are skipped and stay unwritten."""
    workload = _contiguous_workload(_layout("aligned", "per_row", 128), [128, 3])
    a, a_scale, b, b_scale, ids = _scaled_inputs(workload)
    pad = 128

    def padded(t: torch.Tensor, fill) -> torch.Tensor:
        return torch.cat([t, torch.full((pad, *t.shape[1:]), fill, dtype=t.dtype, device=t.device)])

    inputs = (
        padded(a.view(torch.uint8), 0).view(a.dtype),
        padded(a_scale, 1.0),
        b,
        b_scale,
        padded(ids, b.shape[0]),
    )
    out = _run_into_guarded_out(
        MoEGroupedGEMMFP8FwdOp(workload.call.arguments({})["layout"]), workload, *inputs
    )
    assert bool((out[-pad:] == _SENTINEL).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_tiles_divide_the_alignment() -> None:
    """A tile taller than an aligned layout's alignment is refused before tracing."""
    workload = _contiguous_workload(_layout("aligned", "physical_psum", 64), [64, 3])
    config = {"block_m": 128, "block_n": 128, "num_stages": 4, "num_sms": 2, "stage_scales": 0}
    op = _forced_op(workload.call.arguments({})["layout"], config)
    with pytest.raises(ValueError, match="block_m must divide"):
        op(*workload.gen_inputs())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        pytest.param({"block_m": 96}, "block_m must be one of", id="block_m"),
        pytest.param({"block_n": 96}, "block_n must be one of", id="block_n"),
        pytest.param({"num_stages": 2}, "num_stages must lie in", id="stages"),
        pytest.param({"block_m": 128, "block_n": 192, "num_stages": 8}, "shared memory", id="smem"),
        pytest.param({"num_sms": 0}, "num_sms must lie in", id="no-sms"),
        pytest.param({"num_sms": 1 << 16}, "num_sms must lie in", id="past-the-device"),
        pytest.param({"stage_scales": 3}, "stage_scales must be 0, 1 or 2", id="stage-scales"),
        pytest.param({"stage_scales": 2}, "4 K steps at a time", id="tma-scales-k"),
        pytest.param({"threads": 256}, "needs exactly the keys", id="unknown-key"),
    ],
)
def test_fp8_grouped_gemm_refuses_a_config_it_cannot_build(change: dict, reason: str) -> None:
    workload = _workload(64, E=2, N=128, K=128)
    config = {
        "block_m": 64,
        "block_n": 128,
        "num_stages": 4,
        "num_sms": 2,
        "stage_scales": 0,
        **change,
    }
    op = _forced_op(MaskedLayoutSpec(max_m=64), config)
    with pytest.raises(ValueError, match=reason):
        op(*workload.gen_inputs())


@pytest.mark.smoke
def test_fp8_grouped_gemm_config_refusal_covers_the_builder_domain() -> None:
    """Config acceptance matches shared-memory limits, including H200 calibrations."""
    kernel = MoEGroupedGEMMFP8Kernel
    sm_count, smem_limit = 132, 232448 - 1024
    verdicts = []
    for block_m, block_n, stages, staged in itertools.product(
        kernel._BLOCK_M, kernel._BLOCK_N, kernel._NUM_STAGES, (0, 1, 2)
    ):
        config = {
            "block_m": block_m,
            "block_n": block_n,
            "num_stages": stages,
            "num_sms": sm_count,
            "stage_scales": staged,
        }
        accepted = kernel.config_refusal(config, sm_count=sm_count, smem_limit=smem_limit) is None
        assert accepted == (_smem_bytes(block_m, block_n, stages, staged) <= smem_limit), config
        verdicts.append(accepted)
    assert any(verdicts) and not all(verdicts)
    for config in kernel._CONFIGS["h200"].values():
        assert kernel.config_refusal(config, sm_count=sm_count, smem_limit=smem_limit) is None


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
def test_fp8_grouped_gemm_tune_picks_a_correct_config(arch: int) -> None:
    """The tuned kernel handles different expert counts correctly."""
    workload = _workload(128, E=4, N=256, K=512)
    inputs = _with_counts(workload, [128, 0, 33, 100])
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=128))
    op.request_tune()
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    inputs = _with_counts(workload, [7, 128, 0, 64])
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
def test_fp8_grouped_gemm_rejects_another_fp8_format(arch: int) -> None:
    """Reject e5m2 operands before launch."""
    workload = _workload(64, E=2, N=128, K=128)
    a, *rest = workload.gen_inputs()
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=64))
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(a.to(torch.float8_e5m2), *rest)


def _fp8_call(**facts) -> MGroupedGEMMFP8Call:
    return MGroupedGEMMFP8Call(
        **{
            "arch": 90,
            "sm_count": 132,
            "kind": "masked",
            "max_m": 64,
            "ab_dtype": torch.float8_e4m3fn,
            "num_groups": 2,
            "n": 256,
            "k": 256,
            "a_scale_shape": (2, 64, 2),
            "b_scale_shape": (2, 2, 2),
            **facts,
        }
    )


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("facts", "reason"),
    [
        pytest.param(
            {
                "kind": "contiguous",
                "packing": "aligned",
                "metadata_kind": "per_row",
                "alignment": 32,
                "max_m": None,
                "m": 64,
                "a_scale_shape": (64, 2),
            },
            "aligned runs",
            id="aligned-32",
        ),
        pytest.param({"ab_dtype": torch.bfloat16}, "float8_e4m3fn operands", id="bf16"),
        pytest.param({"k": 192, "a_scale_shape": (2, 64, 1)}, "multiples of 128", id="k-off-128"),
        pytest.param({"n": 192, "b_scale_shape": (2, 1, 2)}, "multiples of 128", id="n-off-128"),
        pytest.param({"a_scale_shape": (2, 2)}, "reads a_scale", id="per-tensor-a-scale"),
        pytest.param({"b_scale_shape": (2, 256, 2)}, "reads a_scale", id="per-row-b-scale"),
    ],
)
def test_fp8_grouped_gemm_kernel_refuses_what_it_cannot_run(facts: dict, reason: str) -> None:
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=64))
    with pytest.raises(ValueError, match=reason):
        op.key_for(_KEY, _fp8_call(**facts))


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fp8_grouped_gemm_calibration_yields_to_the_alignment() -> None:
    """An incompatible calibration falls back to an alignment-compatible tile."""
    workload = _contiguous_workload(
        _layout("aligned", "physical_psum", 64), [64, 3] * 8, N=1536, K=2048
    )
    op = MoEGroupedGEMMFP8FwdOp(workload.call.arguments({})["layout"])
    _run_into_guarded_out(op, workload, *workload.gen_inputs())
    (kernel,) = op.iter_kernels()
    assert kernel.config["block_m"] == 64


@pytest.mark.smoke
def test_fp8_grouped_gemm_mma_kernel_rejects_tight_layouts() -> None:
    """The MMA implementation does not serve tight expert runs."""
    call = _fp8_call(
        arch=89,
        kind="contiguous",
        packing="tight",
        metadata_kind="physical_psum",
        max_m=None,
        m=64,
        a_scale_shape=(64, 2),
    )
    assert "masked slabs or aligned per-row" in MoEGroupedGEMMFP8MMAKernel.refusal(call)


# The op's inputs, in the order it takes them.
_FP8_INPUTS = ("a", "a_scale", "b", "b_scale", "layout_metadata")


def _fp8_in_workspace(name, overlap):
    max_m, experts, n, k = 64, 2, 128, 128
    workload = _workload(max_m, E=experts, N=n, K=k)
    inputs = dict(zip(_FP8_INPUTS, _with_counts(workload, [max_m] * experts), strict=True))
    source = inputs[name]
    out_like = torch.empty(experts, max_m, n, dtype=torch.bfloat16)
    workspace = torch.zeros(
        out_like.nbytes + source.nbytes, dtype=torch.uint8, device=source.device
    )
    start = out_like.nbytes - (source.element_size() if overlap else 0)
    inputs[name] = torch.from_dlpack(
        workspace[start : start + source.nbytes].view(source.dtype).view(source.shape).copy_(source)
    )
    out = torch.from_dlpack(workspace[: out_like.nbytes].view(out_like.dtype).view(out_like.shape))
    return MoEGroupedGEMMFP8FwdOp(**workload.call.arguments({})), inputs, out, workload, workspace


@pytest.mark.sm90
@pytest.mark.smoke
def test_fp8_refuses_out_overlapping_any_input():
    """The manifest lets ``out`` overlap no input: an output sharing an element with any input
    is refused without writing."""
    for name in _FP8_INPUTS:
        op, inputs, out, _, workspace = _fp8_in_workspace(name, True)
        before = workspace.clone()
        with pytest.raises(ValueError, match=f"out must not overlap {name}"):
            op(**inputs, out=out)
        assert torch.equal(workspace, before)


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
def test_fp8_accepts_adjacent_dlpack_output(arch: int):
    """An output ending where an input starts is written by each architecture's kernel without
    modifying that input."""
    op, inputs, out, workload, workspace = _fp8_in_workspace("a", False)
    before = workspace.clone()
    assert op(**inputs, out=out) is out
    args = tuple(inputs.values())
    compare_outputs(out, workload.ref_program(*args), workload.verification(*args))
    assert torch.equal(workspace[out.nbytes :], before[out.nbytes :])


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
@pytest.mark.parametrize("offset_inputs", [False, True])
def test_fp8_grouped_gemm_accepts_offset_storage(offset_inputs, arch: int):
    """Offset views preserve input storage, output guards and idle expert rows."""
    workload = _workload(65, E=3, N=256, K=256)
    inputs = list(_with_counts(workload, [0, 33, 65]))
    if offset_inputs:
        for index in (0, 2):
            original = inputs[index]
            shifted = torch.empty(
                original.numel() + 1, dtype=original.dtype, device=original.device
            )[1:].view(original.shape)
            shifted.copy_(original)
            inputs[index] = shifted
            assert shifted.data_ptr() % 16
    before = [x.clone() for x in inputs]
    storage = torch.full(
        (3 * 65 * 256 + 2,), _SENTINEL, dtype=torch.bfloat16, device=inputs[0].device
    )
    out = storage[1:-1].view(3, 65, 256)
    assert out.data_ptr() % 16
    op = MoEGroupedGEMMFP8FwdOp(MaskedLayoutSpec(max_m=65))
    assert op(*inputs, out=out) is out
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    assert bool((out[0] == _SENTINEL).all())
    assert storage[0].item() == _SENTINEL and storage[-1].item() == _SENTINEL
    for old, new in zip(before, inputs, strict=True):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
