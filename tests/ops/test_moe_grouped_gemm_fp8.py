"""Tests for the block-scaled FP8 masked M-grouped GEMM op."""

import itertools

import pytest
import torch

from tileops.backend import register_kernel_type, registry
from tileops.kernels.moe import MGroupedGemmFP8Call, MoEGroupedGemmFP8Kernel
from tileops.kernels.moe.moe_grouped_gemm_fp8 import _smem_bytes
from tileops.manifest import load_workloads
from tileops.ops.moe import MaskedLayoutSpec, MoEGroupedGemmFP8FwdOp
from workloads.moe import MoEGroupedGemmFP8Workload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

_OP = "MoEGroupedGemmFP8FwdOp"
_KEY = "grouped_gemm_fp8"
# An output value no valid row produces, and the elements kept past the output to catch a store
# outside it.
_SENTINEL, _GUARD = -12288.0, 4096


def _workload(max_m: int, **dims) -> MoEGroupedGemmFP8Workload:
    # T only feeds the generated counts; half the slabs' capacity keeps every count in range.
    dims.setdefault("T", dims["E"] * max_m // 2)
    return MoEGroupedGemmFP8Workload(
        manifest_call(_OP, layout={"masked": {"max_m": max_m}}, **dims)
    )


def _with_counts(workload: MoEGroupedGemmFP8Workload, counts: list) -> tuple:
    """The workload's inputs with *counts* valid rows per expert in place of the generated ones."""
    *operands, metadata = workload.gen_inputs()
    return (*operands, torch.tensor(counts, dtype=torch.int32, device=metadata.device))


def _run_into_guarded_out(op, workload: MoEGroupedGemmFP8Workload, *inputs) -> torch.Tensor:
    """Run *op* into a sentinel-filled ``out`` followed by a guard, and check the call.

    ``out=`` is returned, nothing past it is written, and the valid rows match the reference,
    each count read clamped to ``[0, max_m]`` as the op documents.
    """
    a, b, metadata = inputs[0], inputs[2], inputs[-1]
    shape = (*a.shape[:-1], b.shape[1])
    flat = torch.full(
        (a.shape[0] * a.shape[1] * b.shape[1] + _GUARD,),
        _SENTINEL,
        dtype=torch.bfloat16,
        device=a.device,
    )
    out = flat[:-_GUARD].view(shape)
    got = op(*inputs, out=out)
    assert got.data_ptr() == out.data_ptr() and got.shape == shape
    assert bool((flat[-_GUARD:] == _SENTINEL).all()), "a store ran past the output"
    clamped = (*inputs[:-1], metadata.clamp(0, a.shape[1]))
    compare_outputs(got, workload.ref_program(*clamped), workload.verification(*clamped))
    return got


_ROWS = {row["label"]: row for row in load_workloads(_OP)}


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.parametrize(
    "label",
    [
        pytest.param("ds-v3-e8-up-masked", marks=pytest.mark.smoke),
        # A decode shard's slabs: max_m = 64 on the default rule, one 64-row tile per expert;
        # max_m = 256 on its calibrated 192-wide config, four 64-row tiles, most of them empty.
        pytest.param("qwen3-235b-dec-b64-down", marks=pytest.mark.full),
        pytest.param("kimi-k2-dec-b256-up", marks=pytest.mark.full),
        # A generated row with 13 idle experts among 48, spread between the busy ones.
        pytest.param("kimi-k2-dec-b64-down", marks=pytest.mark.full),
    ],
)
def test_fp8_grouped_gemm_runs_the_manifest_rows(label: str, arch: int) -> None:
    """Representative manifest rows, with their generated valid counts, match the reference."""
    row = {key: value for key, value in _ROWS[label].items() if key != "label"}
    workload = MoEGroupedGemmFP8Workload(manifest_call(_OP, **row))
    inputs = workload.gen_inputs()
    out = MoEGroupedGemmFP8FwdOp(**workload.call.arguments({}))(*inputs)
    assert out.dtype is torch.bfloat16
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("max_m", "dims", "counts"),
    [
        # Shape: one 64-row tile per slab, at the smallest N and K the scales admit (one b_scale
        # row, one K step); counts max_m, 0, 1 and one past max_m on the last slab.
        pytest.param(64, {"N": 128, "K": 128}, [64, 0, 1, 99], id="max_m-64-n128-k128"),
        # Shape: the first max_m that takes 128-row tiles, its one tile past max_m.
        pytest.param(65, {"K": 256}, [65, 0, 64, 1], id="max_m-65"),
        # Shape: a slab of exactly one 128-row tile; a negative count reads as 0, so the
        # experts after it keep their tiles.
        pytest.param(128, {}, [128, -1000, 1, 0], id="max_m-128"),
        # Shape: the first max_m back on 64-row tiles, three per slab with the last holding one
        # row; counts 0, 1, block_m, block_m + 1, max_m and past it.
        pytest.param(129, {}, [0, 1, 64, 65, 129, 200], id="max_m-129"),
        # Shape: max_m no multiple of 64; the last slab's full tile ends past the output, so the
        # guard sees a store not clipped at max_m.
        pytest.param(100, {}, [100, 0, 100], id="max_m-100"),
        # Shape: one expert of 18 M tiles, past the 16 one schedule group takes, so the general
        # group schedule runs with a second group of two; two N tiles, so a group's M and N
        # tiles interleave.
        pytest.param(1152, {"K": 128}, [1100], id="max_m-1152-e1"),
        # Shape: empty slabs; the op returns out without a launch.
        pytest.param(0, {"N": 128, "K": 128}, [0, 0], id="max_m-0"),
    ],
)
def test_fp8_grouped_gemm_valid_rows_at_tile_and_slab_boundaries(
    max_m: int, dims: dict, counts: list, arch: int
) -> None:
    dims = {"N": 256, "K": 512, **dims}
    workload = _workload(max_m, E=len(counts), **dims)
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=max_m))
    _run_into_guarded_out(op, workload, *_with_counts(workload, counts))


def _forced_op(max_m: int, config: dict) -> MoEGroupedGemmFP8FwdOp:
    """An op whose kernel runs *config*: the in-tree kernel under it, registered under its own
    key and preferred over the in-tree one."""

    class Forced(MoEGroupedGemmFP8Kernel):
        preferred_over = frozenset(MoEGroupedGemmFP8FwdOp.kernel_types)

        @property
        def default_config(self) -> dict:
            return dict(config)

    state = registry.snapshot()
    try:
        register_kernel_type(_OP, f"{_KEY}_forced", Forced)
        return MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=max_m))
    finally:
        registry.restore(state)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("block_n", "num_stages", "n"),
    [
        # Feature: a 160-wide tile at every offset n0 % 128 takes (0, 32, 64, 96), so each of
        # its column sub-blocks reads its own b scale row; the last tile holds 32 of its
        # columns. Four stages divide the four K steps: the constant-slot ring.
        pytest.param(160, 4, 512, id="bn160-static-ring"),
        # Feature: a 192-wide tile, the last holding 128 of its columns. Three stages do not
        # divide the K steps: the ring position carries across a block's tasks.
        pytest.param(192, 3, 512, id="bn192-index-ring"),
        # Shape: two 192-wide N tiles and no N tail; the first tile's middle sub-block lies in
        # b scale row 0, the second tile's first in row 1.
        pytest.param(192, 4, 384, id="bn192-two-tiles-n384"),
        # Shape: two 192-wide N tiles, the last holding 64 columns, so its sub-blocks past N
        # clamp to the last b scale row.
        pytest.param(192, 4, 256, id="bn192-two-tiles-n256"),
        # Shape: two 160-wide N tiles, the last holding 96 columns.
        pytest.param(160, 4, 256, id="bn160-two-tiles-n256"),
    ],
)
def test_fp8_grouped_gemm_wide_tiles_clip_at_n(block_n: int, num_stages: int, n: int) -> None:
    """Tiles past N, over a slab whose last 128-row tile runs past max_m = 192.

    Three blocks run every tile, so each reuses its staged tile. The rows of the empty expert
    and of the skipped M tile stay untouched: the store neither crosses into the next slab nor
    runs for a tile with no valid row.
    """
    counts = [192, 0, 70]
    workload = _workload(192, E=len(counts), N=n, K=512)
    config = {"block_m": 128, "block_n": block_n, "num_stages": num_stages, "num_sms": 3}
    op = _forced_op(192, config)
    out = _run_into_guarded_out(op, workload, *_with_counts(workload, counts))
    assert bool((out[1] == _SENTINEL).all()) and bool((out[2, 128:] == _SENTINEL).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        # Feature: each constraint of config_refusal, met before anything is traced.
        pytest.param({"block_m": 96}, "block_m must be one of", id="block_m"),
        pytest.param({"block_n": 96}, "block_n must be one of", id="block_n"),
        pytest.param({"num_stages": 2}, "num_stages must lie in", id="stages"),
        pytest.param({"block_m": 128, "block_n": 192, "num_stages": 8}, "shared memory", id="smem"),
        pytest.param({"num_sms": 0}, "num_sms must lie in", id="no-sms"),
        pytest.param({"num_sms": 1 << 16}, "num_sms must lie in", id="past-the-device"),
        pytest.param({"threads": 256}, "needs exactly the keys", id="unknown-key"),
    ],
)
def test_fp8_grouped_gemm_refuses_a_config_it_cannot_build(change: dict, reason: str) -> None:
    workload = _workload(64, E=2, N=128, K=128)
    config = {"block_m": 64, "block_n": 128, "num_stages": 4, "num_sms": 2, **change}
    op = _forced_op(64, config)
    with pytest.raises(ValueError, match=reason):
        op(*workload.gen_inputs())


@pytest.mark.smoke
def test_fp8_grouped_gemm_config_refusal_covers_the_builder_domain() -> None:
    """Compile-free: over the whole product of tile widths and stage counts a config is accepted
    exactly when its ring fits the block's shared memory, and every calibrated entry fits H200."""
    kernel = MoEGroupedGemmFP8Kernel
    sm_count, smem_limit = 132, 232448 - 1024
    verdicts = []
    for block_m, block_n, stages in itertools.product(
        kernel._BLOCK_M, kernel._BLOCK_N, kernel._NUM_STAGES
    ):
        config = {"block_m": block_m, "block_n": block_n, "num_stages": stages, "num_sms": sm_count}
        accepted = kernel.config_refusal(config, sm_count=sm_count, smem_limit=smem_limit) is None
        assert accepted == (_smem_bytes(block_m, block_n, stages) <= smem_limit), config
        verdicts.append(accepted)
    assert any(verdicts) and not all(verdicts)
    for config in kernel._CONFIGS["h200"].values():
        assert kernel.config_refusal(config, sm_count=sm_count, smem_limit=smem_limit) is None


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
def test_fp8_grouped_gemm_tune_picks_a_correct_config(arch: int) -> None:
    """Feature: ``request_tune()`` sweeps the candidates; the chosen one still matches the
    reference."""
    workload = _workload(128, E=4, N=256, K=512)
    inputs = _with_counts(workload, [128, 0, 33, 100])
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=128))
    op.request_tune()
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    # A second call after the sweep runs the chosen config on other counts, with no state
    # left over from tuning.
    inputs = _with_counts(workload, [7, 128, 0, 64])
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
def test_fp8_grouped_gemm_rejects_another_fp8_format(arch: int) -> None:
    """Dtype: the signature takes only e4m3 operands; an e5m2 ``a`` is refused before any
    kernel."""
    workload = _workload(64, E=2, N=128, K=128)
    a, *rest = workload.gen_inputs()
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=64))
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(a.to(torch.float8_e5m2), *rest)


def _fp8_call(**facts) -> MGroupedGemmFP8Call:
    return MGroupedGemmFP8Call(
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
        # Feature: each refusal of the kernel, for a call the signature would reject first.
        pytest.param({"kind": "contiguous"}, "only the masked layout", id="contiguous"),
        pytest.param({"ab_dtype": torch.bfloat16}, "float8_e4m3fn operands", id="bf16"),
        pytest.param({"k": 192, "a_scale_shape": (2, 64, 1)}, "multiples of 128", id="k-off-128"),
        pytest.param({"n": 192, "b_scale_shape": (2, 1, 2)}, "multiples of 128", id="n-off-128"),
        pytest.param({"a_scale_shape": (2, 2)}, "reads a_scale", id="per-tensor-a-scale"),
        pytest.param({"b_scale_shape": (2, 256, 2)}, "reads a_scale", id="per-row-b-scale"),
    ],
)
def test_fp8_grouped_gemm_kernel_refuses_what_it_cannot_run(facts: dict, reason: str) -> None:
    op = MoEGroupedGemmFP8FwdOp(MaskedLayoutSpec(max_m=64))
    with pytest.raises(ValueError, match=reason):
        op.key_for(_KEY, _fp8_call(**facts))


def _fp8_in_workspace(name, overlap):
    max_m, experts, n, k = 64, 2, 128, 128
    workload = _workload(max_m, E=experts, N=n, K=k)
    names = ("a", "a_scale", "b", "b_scale", "layout_metadata")
    inputs = dict(zip(names, _with_counts(workload, [max_m] * experts), strict=True))
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
    return MoEGroupedGemmFP8FwdOp(**workload.call.arguments({})), inputs, out, workload, workspace


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
@pytest.mark.parametrize("name", ["a", "a_scale", "b", "b_scale", "layout_metadata"])
def test_fp8_refuses_overlapping_dlpack_output(name, arch: int):
    """An output sharing an element with any input is refused without writing."""
    op, inputs, out, _, workspace = _fp8_in_workspace(name, True)
    before = workspace.clone()
    with pytest.raises(ValueError, match=f"out must not overlap {name}"):
        op(**inputs, out=out)
    assert torch.equal(workspace, before)


@pytest.mark.parametrize(
    "arch", [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)]
)
@pytest.mark.smoke
@pytest.mark.parametrize("name", ["a", "a_scale", "b", "b_scale", "layout_metadata"])
def test_fp8_accepts_adjacent_dlpack_output(name, arch: int):
    """An output ending where an input starts is written without modifying that input."""
    op, inputs, out, workload, workspace = _fp8_in_workspace(name, False)
    before = workspace.clone()
    assert op(**inputs, out=out) is out
    args = tuple(inputs.values())
    compare_outputs(out, workload.ref_program(*args), workload.verification(*args))
    assert torch.equal(workspace[out.nbytes :], before[out.nbytes :])
