"""Benchmarks for the RoPE op family.

Workload shapes, dtypes, layouts, and roofline formulas are loaded from the
ops manifest (``tileops/manifest/position_encoding.yaml``); nothing about a
workload is hard-coded here.

One ``test_*_bench`` per op, so the validator's L4 AST check can tie each
``load_workloads("<OpName>")`` call to its manifest entry.

Baselines build their cos/sin tables outside the timed window, so only the
rotation itself is measured.
"""

import math

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.rope import (
    RopeLlama31Op,
    RopeLongRopeOp,
    RopeNeoxOp,
    RopeNeoxPositionIdsOp,
    RopeNonNeoxOp,
    RopeYarnOp,
)

# Bench-local: manifest workload entries carry no ``base``; the ops and the
# baseline both use the manifest signature default (``base: 10000.0``).
_BASE = 10000.0


class _RopeWorkload:
    """Minimal :class:`ShapeDtypeWorkload` for the RoPE family.

    Holds ``shape`` and ``dtype`` so :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _mark(idx: int):
    """First manifest workload of an op is the smoke case; the rest are full."""
    return pytest.mark.smoke if idx == 0 else pytest.mark.full


def _layout_params(workloads: list[dict]) -> list:
    """Build ``(shape, dtype, layout)`` params for the 1d/2d RoPE variants."""
    params = []
    for idx, w in enumerate(workloads):
        layout = w["layout"]
        if layout == "1d":
            shape = (w["seq_len"], w["head_dim"])
        else:
            shape = (w["batch"], w["seq_len"], w["num_heads"], w["head_dim"])
        for dtype_name in w["dtypes"]:
            params.append(pytest.param(
                shape, getattr(torch, dtype_name), layout,
                id=f"{w['label']}-{dtype_name}",
                marks=_mark(idx),
            ))
    return params


def _position_ids_params(workloads: list[dict]) -> list:
    """Build ``(shape, dtype, max_position)`` params for the THD variant."""
    params = []
    for idx, w in enumerate(workloads):
        shape = (w["num_tokens"], w["num_heads"], w["head_dim"])
        for dtype_name in w["dtypes"]:
            params.append(pytest.param(
                shape, getattr(torch, dtype_name), w["max_position"],
                id=f"{w['label']}-{dtype_name}",
                marks=_mark(idx),
            ))
    return params


# Bench-local PyTorch baselines


def _rope_tables(seq_len: int, head_dim: int, dtype: torch.dtype):
    cos, sin = _rope_half_tables(seq_len, head_dim)
    return torch.cat([cos] * 2, dim=-1).to(dtype), torch.cat([sin] * 2, dim=-1).to(dtype)


def _rope_half_tables(
    seq_len: int,
    head_dim: int,
    *,
    variant: str = "neox",
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    dim = torch.arange(0, half, device="cuda", dtype=torch.float32)
    if variant == "llama31":
        freqs = _llama31_inv_freqs(dim, half)
    elif variant == "yarn":
        freqs = _yarn_inv_freqs(dim, half)
    elif variant == "longrope":
        freqs = _longrope_inv_freqs(dim, half)
    else:
        freqs = 1.0 / (_BASE ** (dim / half))
    angles = torch.outer(torch.arange(seq_len, device="cuda", dtype=torch.float32), freqs)
    return torch.cos(angles), torch.sin(angles)


def _llama31_inv_freqs(dim: torch.Tensor, half: int) -> torch.Tensor:
    freqs = 1.0 / (_BASE ** (dim / half))
    wavelen = 2 * math.pi / freqs
    low_freq_wavelen = 8192.0
    high_freq_wavelen = 8192.0 / 4.0
    smooth = (8192.0 / wavelen - 1.0) / 3.0
    blended = (1.0 - smooth) * freqs / 8.0 + smooth * freqs
    return torch.where(
        wavelen < high_freq_wavelen,
        freqs,
        torch.where(wavelen > low_freq_wavelen, freqs / 8.0, blended),
    )


def _yarn_find_correction_dim(num_rotations: float, dim: int) -> float:
    return dim * math.log(4096 / (num_rotations * 2 * math.pi)) / (2 * math.log(_BASE))


def _yarn_inv_freqs(dim: torch.Tensor, half: int) -> torch.Tensor:
    beta_fast = 32.0
    beta_slow = 1.0
    scale = 16.0
    low = max(math.floor(_yarn_find_correction_dim(beta_fast, half)), 0)
    high = min(math.ceil(_yarn_find_correction_dim(beta_slow, half)), half - 1)
    if low == high:
        high += 1
    freq_extra = 1.0 / (_BASE ** (dim / half))
    freq_inter = 1.0 / ((scale * _BASE) ** (dim / half))
    inv_freq_mask = 1.0 - torch.clamp((dim - low) / (high - low), 0.0, 1.0)
    return freq_inter * (1.0 - inv_freq_mask) + freq_extra * inv_freq_mask


def _longrope_inv_freqs(dim: torch.Tensor, half: int) -> torch.Tensor:
    # Manifest workloads use RopeLongRopeOp defaults: no rescale factors and
    # equal max/original position lengths, so the scale amplitude remains 1.
    return 1.0 / (_BASE ** (dim / half))


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _rotate_interleaved(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_odd * cos + x_even * sin
    return out


def _rope_reference(
    x: torch.Tensor,
    layout: str,
    *,
    variant: str,
    interleave: bool = False,
) -> torch.Tensor:
    seq_len = x.shape[0] if layout == "1d" else x.shape[1]
    cos_half, sin_half = _rope_half_tables(seq_len, x.shape[-1], variant=variant)
    if interleave:
        if layout != "1d":
            cos_half, sin_half = (
                t.view(1, seq_len, 1, x.shape[-1] // 2) for t in (cos_half, sin_half)
            )
        return _rotate_interleaved(x, cos_half.to(x.dtype), sin_half.to(x.dtype))

    cos = torch.cat([cos_half] * 2, dim=-1).to(x.dtype)
    sin = torch.cat([sin_half] * 2, dim=-1).to(x.dtype)
    if layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, x.shape[-1]) for t in (cos, sin))
    return _rotate(x, cos, sin)


def _assert_flashinfer_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def _flashinfer_module():
    try:
        import flashinfer
    except ImportError:
        return None
    return flashinfer


def _flashinfer_qk_pos_ids(
    x: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    if layout == "1d":
        seq_len, head_dim = x.shape
        q = x.view(seq_len, 1, head_dim)
        pos_ids = torch.arange(seq_len, device=x.device, dtype=torch.int32)
    else:
        batch, seq_len, num_heads, head_dim = x.shape
        q = x.reshape(batch * seq_len, num_heads, head_dim)
        pos_ids = torch.arange(
            seq_len, device=x.device, dtype=torch.int32,
        ).repeat(batch)
    k = x.new_empty((q.shape[0], 0, q.shape[-1]))
    return q, k, pos_ids, tuple(x.shape)


def _flashinfer_flat_qk_pos_ids(
    x: torch.Tensor,
    layout: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    q, k, pos_ids, output_shape = _flashinfer_qk_pos_ids(x, layout)
    return q.reshape(q.shape[0], -1), k.reshape(k.shape[0], -1), pos_ids, output_shape


def _rope_cos_sin_cache(
    seq_len: int,
    head_dim: int,
    *,
    variant: str = "neox",
) -> torch.Tensor:
    cos, sin = _rope_half_tables(seq_len, head_dim, variant=variant)
    return torch.cat([cos, sin], dim=-1)


def _record_flashinfer_rope(
    flashinfer,
    op,
    bm: ManifestBenchmark,
    params: dict,
    x: torch.Tensor,
    layout: str,
    *,
    variant: str,
    interleave: bool = False,
) -> None:
    q, k, pos_ids, output_shape = _flashinfer_qk_pos_ids(x, layout)
    if variant == "llama31":
        def flashinfer_baseline(q, k, p):
            return flashinfer.apply_llama31_rope_pos_ids(
                q,
                k,
                p,
                interleave=interleave,
                rope_theta=_BASE,
            )[0].reshape(output_shape)
    else:
        def flashinfer_baseline(q, k, p):
            return flashinfer.apply_rope_pos_ids(
                q,
                k,
                p,
                interleave=interleave,
                rope_theta=_BASE,
            )[0].reshape(output_shape)

    _assert_flashinfer_close(
        flashinfer_baseline(q, k, pos_ids),
        _rope_reference(x, layout, variant=variant, interleave=interleave),
    )
    result_fi = bm.profile(flashinfer_baseline, q, k, pos_ids)
    BenchmarkReport.record(op, params, result_fi, tag="flashinfer")


def _record_flashinfer_cached_rope(
    flashinfer,
    op,
    bm: ManifestBenchmark,
    params: dict,
    x: torch.Tensor,
    layout: str,
    *,
    variant: str,
    is_neox: bool = True,
) -> None:
    seq_len = x.shape[0] if layout == "1d" else x.shape[1]
    q, k, pos_ids, output_shape = _flashinfer_flat_qk_pos_ids(x, layout)
    cos_sin_cache = _rope_cos_sin_cache(seq_len, x.shape[-1], variant=variant)

    def flashinfer_baseline(p, q, k):
        return flashinfer.apply_rope_with_cos_sin_cache(
            p,
            q,
            k,
            x.shape[-1],
            cos_sin_cache,
            is_neox=is_neox,
        )[0].reshape(output_shape)

    _assert_flashinfer_close(
        flashinfer_baseline(pos_ids, q, k),
        _rope_reference(x, layout, variant=variant, interleave=(not is_neox)),
    )
    result_fi = bm.profile(flashinfer_baseline, pos_ids, q, k)
    BenchmarkReport.record(op, params, result_fi, tag="flashinfer")


def _profile_rope(op, bm: ManifestBenchmark, shape: tuple[int, ...],
                  dtype: torch.dtype, layout: str, *, variant: str,
                  interleave: bool = False) -> None:
    """Profile op and the torch rotation baseline on the same input."""
    x = torch.randn(shape, device="cuda", dtype=dtype)
    params = {"shape": shape, "dtype": dtype, "layout": layout}

    result = bm.profile(op, x)
    BenchmarkReport.record(op, params, result, tag="tileops")

    flashinfer = _flashinfer_module()
    if flashinfer is not None:
        if variant in ("neox", "non_neox", "llama31"):
            _record_flashinfer_rope(
                flashinfer, op, bm, params, x, layout,
                variant=variant, interleave=interleave,
            )
        else:
            _record_flashinfer_cached_rope(
                flashinfer, op, bm, params, x, layout,
                variant=variant,
                is_neox=(not interleave),
            )

    result_bl = bm.profile(
        lambda t: _rope_reference(t, layout, variant=variant, interleave=interleave),
        x,
    )
    BenchmarkReport.record(op, params, result_bl, tag="torch_ref")


# Per-op tests — one block per manifest entry.

_NEOX_OP = "RopeNeoxOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_NEOX_OP)),
)
def test_rope_neox_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeNeoxOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_NEOX_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, variant="neox")


_NON_NEOX_OP = "RopeNonNeoxOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_NON_NEOX_OP)),
)
def test_rope_non_neox_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeNonNeoxOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_NON_NEOX_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, variant="non_neox", interleave=True)


_LLAMA31_OP = "RopeLlama31Op"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_LLAMA31_OP)),
)
def test_rope_llama31_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeLlama31Op(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_LLAMA31_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, variant="llama31")


_YARN_OP = "RopeYarnOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_YARN_OP)),
)
def test_rope_yarn_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeYarnOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_YARN_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, variant="yarn")


_LONGROPE_OP = "RopeLongRopeOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_LONGROPE_OP)),
)
def test_rope_longrope_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeLongRopeOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_LONGROPE_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, variant="longrope")


_POSITION_IDS_OP = "RopeNeoxPositionIdsOp"


@pytest.mark.parametrize(
    "shape, dtype, max_position",
    _position_ids_params(load_workloads(_POSITION_IDS_OP)),
)
def test_rope_neox_position_ids_bench(
    shape: tuple[int, int, int], dtype: torch.dtype, max_position: int,
) -> None:
    num_tokens, _, head_dim = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    position_ids = torch.arange(
        num_tokens, device="cuda", dtype=torch.int32,
    ) % max_position

    op = RopeNeoxPositionIdsOp(max_position=max_position, base=_BASE)
    bm = ManifestBenchmark(_POSITION_IDS_OP, op, _RopeWorkload(shape, dtype))
    params = {"shape": shape, "dtype": dtype, "max_position": max_position}

    result = bm.profile(op, x, position_ids)
    BenchmarkReport.record(op, params, result, tag="tileops")

    flashinfer = _flashinfer_module()
    if flashinfer is not None:
        k = x.new_empty((num_tokens, 0, head_dim))
        def flashinfer_baseline(t, k, p):
            return flashinfer.apply_rope_pos_ids(
                t,
                k,
                p,
                rope_theta=_BASE,
            )[0]

        cos, sin = _rope_tables(max_position, head_dim, dtype)

        def expected_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
            idx = pos.long()
            return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

        _assert_flashinfer_close(flashinfer_baseline(x, k, position_ids), expected_fn(x, position_ids))
        result_fi = bm.profile(flashinfer_baseline, x, k, position_ids)
        BenchmarkReport.record(op, params, result_fi, tag="flashinfer")

    cos, sin = _rope_tables(max_position, head_dim, dtype)

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    result_bl = bm.profile(baseline_fn, x, position_ids)
    BenchmarkReport.record(op, params, result_bl, tag="torch_ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
