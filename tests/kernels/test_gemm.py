"""GEMM kernels: default configs, autotune spaces, and the configs they refuse."""

import pytest
import torch

from tileops.kernels.gemm import GemmCpAsyncKernel, GemmTMAKernel, GemmW4A16Kernel, GemvKernel
from tileops.kernels.gemm.call_spec import GemmCall
from tileops.kernels.gemm.dense import GemmFP8BlockScaleKernel, _bandwidth_autotune_grid
from tileops.kernels.gemm.fp8_1d2d import GemmFP81D2DFwdKernel
from tileops.kernels.gemm.heuristics import gemv_config, small_batch_config
from workloads.gemm import GemmFP8Workload


def _gemm_call(m: int, n: int, k: int, *, dtype=torch.float16, trans_b: bool = True) -> GemmCall:
    """One dense GEMM call on the SM90 board the regions were fitted on."""
    return GemmCall(arch=90, sm_count=132, m=m, n=n, k=k, dtype=dtype, trans_b=trans_b)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        pytest.param(
            (4096, 2112, 7168),
            (128, 4),
            marks=pytest.mark.smoke,
            id="prefill-qkv-a",
        ),
        pytest.param(
            (128, 7168, 2048),
            (64, 5),
            marks=pytest.mark.full,
            id="decode-grid-underfills",
        ),
        pytest.param(
            (8, 7168, 2048),
            (128, 6),
            marks=pytest.mark.full,
            id="tiny-m-widens-the-tile",
        ),
        pytest.param(
            (4096, 7168, 16384),
            (128, 3),
            marks=pytest.mark.full,
            id="long-k-shallow-ring",
        ),
    ],
)
def test_gemm_fp8_block128_default_config(
    shape: tuple[int, int, int], expected: tuple[int, int]
) -> None:
    kernel = GemmFP8BlockScaleKernel(
        *shape,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
    )

    assert (kernel.config["block_n"], kernel.config["num_stages"]) == expected


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_fp8_1d2d_refuses_a_block_n_that_splits_a_scale_block() -> None:
    """A ``block_n`` that does not divide 128 would leave STSM columns unwritten."""
    kernel = GemmFP81D2DFwdKernel(
        128,
        256,
        512,
        torch.float8_e4m3fn,
        torch.bfloat16,
        config={"block_n": 56, "num_stages": 3, "group_size_m": 16, "group_unroll": 1},
    )
    inputs = GemmFP8Workload(128, 256, 512, torch.float8_e4m3fn, "block128x128").gen_inputs()
    with pytest.raises(ValueError, match="block_n must be one of"):
        kernel(*inputs)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemv_bands_build_their_own_body_and_config() -> None:
    """Each band states its own body shape and config band; the band is in the identity.

    The three bands share ``_gemm_small_batch_kernel``, so what separates them is what
    this asserts: how many rows the body contracts, which config rule picks its
    parameters, and that two bands never share a cache entry.
    """
    from tileops.utils import get_sm_count

    fp = torch.float16

    rows_identity, _ = GemvKernel.entry_for(_gemm_call(2, 2112, 7168))
    row_identity, _ = GemvKernel.entry_for(_gemm_call(1, 2112, 7168))
    col_identity, _ = GemvKernel.entry_for(_gemm_call(2112, 1, 7168, trans_b=False))
    assert rows_identity[0] == "lhs_rows"
    assert row_identity[0] == "lhs_row"
    assert col_identity[0] == "rhs_col"
    assert len({rows_identity, row_identity, col_identity}) == 3

    rows = GemvKernel("lhs_rows", 2, 2112, 7168, fp)
    row = GemvKernel("lhs_row", 1, 2112, 7168, fp)
    col = GemvKernel("rhs_col", 2112, 1, 7168, fp)
    assert (rows.out_len, row.out_len, col.out_len) == (2112, 2112, 2112)
    assert rows.default_config == small_batch_config(2112, 7168, get_sm_count())
    assert row.default_config == gemv_config(7168) == col.default_config
    assert rows.autotune_configs == _bandwidth_autotune_grid((32, 64, 128), (1, 2, 4), (2, 3, 4, 5))
    assert row.autotune_configs == _bandwidth_autotune_grid(
        (32, 64, 128, 256), (1, 2, 4, 8, 16), (1, 2, 3, 4, 5, 6)
    )
    assert col.autotune_configs == row.autotune_configs

    with pytest.raises(ValueError, match="serves bands"):
        GemvKernel("m2", 2, 2112, 7168, fp)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_explicit_structure_config_is_taken_verbatim() -> None:
    """A structure-flagged ``config=`` survives instead of being merged away.

    ``GemmTMAKernel`` has one config schema per structure, so the base's
    merge-over-``default_config`` would drop the caller's flag and keep their tile
    values — asking for ``coop2s`` on a shape the selector serves with ``coop2``
    yielded ``coop2`` at ``coop2s``' ``block_n``, which no measurement covers.
    """
    assert GemmTMAKernel(1536, 2112, 256, torch.bfloat16, trans_b=True).config["block_n"] == 192

    requested = {"coop2s": True, "block_n": 64, "block_k": 128, "num_stages": 4}
    kernel = GemmTMAKernel(1536, 2112, 256, torch.bfloat16, trans_b=True, config=dict(requested))
    assert kernel.config == requested

    merged = GemmTMAKernel(512, 512, 512, torch.float16, config={"block_k": 32}).config
    assert merged["block_k"] == 32
    assert "block_m" in merged and "panel_size" in merged


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.sm90
def test_gemm_tma_kernel_tune_falls_back_to_default() -> None:
    """``GemmTMAKernel`` defines no ``autotune_configs``: ``tune=True`` must warn
    and fall back to ``default_config``.

    The in-tree tuner sweeps only the basic mainloop builder, so a silent
    basic-grid sweep would downgrade shapes whose default is a structure-
    flagged config (coop2 / split-K). Construction only — no JIT compile.
    """
    with pytest.warns(UserWarning, match="does not define autotune_configs"):
        kernel = GemmTMAKernel(
            4096, 4096, 7168, torch.float16, tune=True, trans_a=False, trans_b=True
        )
    assert kernel.config == kernel.default_config


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_slices_k_only_where_the_grid_underfills() -> None:
    assert GemmW4A16Kernel(1, 1024, 8192, torch.float16).config["split_k"] > 1
    assert GemmW4A16Kernel(1, 8192, 8192, torch.float16).config["split_k"] == 1


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_stream_k_compiles_exact_two_way_partition() -> None:
    """The 132-CTA default compiles when 66 N tiles divide into exactly two K slices each."""
    kernel = GemmW4A16Kernel(1, 4224, 81920, torch.float16)
    assert kernel.config["stream_ctas"] == 2 * (4224 // kernel.config["block_n"])
    kernel.kernel(**kernel.config)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gemm_w4a16_stream_k_refuses_an_out_of_range_cta_count() -> None:
    m, n, k = 1, 2176, 32896
    tile = {
        "block_m": 8,
        "block_n": 64,
        "block_k": 512,
        "num_stages": 2,
        "threads": 128,
        "producer_reg": 0,
        "consumer_reg": 0,
        "split_k": 1,
    }
    for stream_ctas in (-1, 1, 34, 69):
        invalid = GemmW4A16Kernel(
            m, n, k, torch.float16, config={**tile, "stream_ctas": stream_ctas}
        )
        with pytest.raises(ValueError, match="stream_ctas"):
            invalid.kernel(**invalid.config)


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.smoke
def test_dense_kernels_refuse_a_shape_they_cannot_load() -> None:
    """Built directly on a shape selection refuses, each mainloop names what it cannot load."""
    with pytest.raises(ValueError, match="cannot serve k=1"):
        GemmCpAsyncKernel(64, 64, 1, torch.float16, trans_b=True)
    with pytest.raises(ValueError, match=r"cannot serve 256x512x1001"):
        GemmTMAKernel(256, 512, 1001, torch.bfloat16, trans_a=False, trans_b=True)
