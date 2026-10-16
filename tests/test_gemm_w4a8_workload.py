"""The W4A8 workload: its reference weight order, and its verifier held to ``GemmW4A8FwdOp``.

``repack_w4a8_weight`` is checked nibble by nibble against the layout its docstring states.
The op leaves one decision open, how the FP8 products are accumulated over K. A positive
control accumulates them another way and must pass. A negative control breaks one step the
op fixes, keeps the rest exact and must be rejected: the weight is not rounded to e4m3, a
scale is dropped or read from the wrong group, an INT4 is read unsigned, or one output is
wrong. A differential test compares the verifier with an oracle written from the op's
docstring.

Tensors stay on the host and no kernel runs: the verifier is fed constructed outputs.
"""

import pytest
import torch

from workloads.gemm import (
    GemmW4A8Workload,
    repack_w4a8_weight,
    unpack_signed_int4,
    unrepack_w4a8_weight,
)
from workloads.numerics import compare_outputs

M, N, K, GROUP = 4, 64, 2048, 128


@pytest.mark.smoke
def test_repack_w4a8_weight_places_each_nibble() -> None:
    """Every nibble lands where the layout formula puts it, encoded, and comes back."""
    n, k = 112, 384  # one full 64-row block and three 16-row units past it
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
    q = unpack_signed_int4(packed)

    prepacked = repack_w4a8_weight(packed)

    row = torch.arange(n)[:, None].expand(n, k)
    col = torch.arange(k)[None, :].expand(n, k)
    block, units = row // 64, torch.where(row // 64 < n // 64, 4, (n % 64) // 16)
    group, half, step, slot, t, i = (
        col // 128,
        col // 64 % 2,
        col // 32 % 2,
        col // 16 % 2,
        col // 4 % 4,
        col % 4,
    )
    offset = (
        block * 64 * k // 2
        + (group * 2 + half) * units * 512
        + (row % 64 // 16) * 512
        + (4 * (row % 8) + t) * 16
        + (2 * step + row % 16 // 8) * 4
        + (4 * slot + i) // 2
    )
    nibble = (prepacked.reshape(-1)[offset].to(torch.int16) >> (4 * (i % 2))) & 0xF
    code = torch.where(q > 0, q - 1, torch.where(q == 0, 7, 8 | (-q - 1)))
    assert torch.equal(nibble, code)
    assert torch.equal(unrepack_w4a8_weight(prepacked), packed)


def _inputs(seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    activation = (torch.randn(M, K, generator=generator) * 0.25).to(torch.float8_e4m3fn)
    q = torch.randint(-8, 8, (N, K), generator=generator, dtype=torch.int16)
    packed = ((q[:, 0::2] & 0xF) | ((q[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    group_scale = (0.03 + 0.05 * torch.rand(N, K // GROUP, generator=generator)).to(
        torch.float8_e4m3fn
    )
    channel_scale = 0.5 + torch.rand(N, generator=generator)
    token_scale = 0.5 + torch.rand(M, generator=generator)
    return activation, packed, group_scale, channel_scale, token_scale


def _products(inputs, *, round_weight=True, signed=True, group_shift=0) -> torch.Tensor:
    """``[M, N, K]`` products of the activation and the weight, each exact in float32."""
    activation, packed, group_scale, _, _ = inputs
    q = unpack_signed_int4(packed).float()
    if not signed:
        q = torch.where(q < 0, q + 16, q)
    scale = group_scale.float().roll(-group_shift, dims=1).repeat_interleave(GROUP, dim=1)
    weight = q * scale
    if round_weight:
        weight = weight.to(torch.float8_e4m3fn).float()
    return activation.float()[:, None, :] * weight[None, :, :]


def _finish(acc: torch.Tensor, inputs, *, channel=True, token=True) -> torch.Tensor:
    _, _, _, channel_scale, token_scale = inputs
    out = acc.double()
    if token:
        out = out * token_scale.double()[:, None]
    if channel:
        out = out * channel_scale.double()[None, :]
    return out.to(torch.bfloat16)


def _truncated_sum(products: torch.Tensor, mantissa_bits: int, chunk: int) -> torch.Tensor:
    """Sum over K a ``chunk`` at a time into an accumulator that keeps ``mantissa_bits``."""
    acc = torch.zeros(products.shape[:-1], dtype=torch.float32)
    drop = 23 - mantissa_bits
    for start in range(0, products.shape[-1], chunk):
        acc = acc + products[..., start : start + chunk].sum(-1)
        bits = acc.view(torch.int32)
        acc = (bits & ~((1 << drop) - 1)).view(torch.float32)
    return acc


def _accepts(inputs, got: torch.Tensor) -> bool:
    workload = GemmW4A8Workload(M, N, K, GROUP)
    try:
        compare_outputs(got, workload.ref_program(*inputs), workload.verification(*inputs))
    except AssertionError:
        return False
    return True


def _docstring_oracle(inputs, got: torch.Tensor) -> bool:
    """Whether *got* is a result ``GemmW4A8FwdOp``'s docstring allows, read literally."""
    r = _finish(_products(inputs).double().sum(-1), inputs).double()
    o = got.double()
    if got.dtype != torch.bfloat16 or got.shape != (M, N):
        return False
    within = (o - r).abs() <= 5e-5 * K + 2e-2 * r.abs()
    energy = (o * o + r * r).sum()
    return bool(within.all()) and bool(((o - r) ** 2).sum() <= 5e-5 * energy)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "accumulate",
    [
        pytest.param(lambda p: p.flip(-1).sum(-1), id="k-reversed"),
        pytest.param(lambda p: p.reshape(M, N, 8, -1).sum(-1).sum(-1), id="split-k-8"),
        # 13 mantissa bits kept after every 128-wide K block, truncated.
        pytest.param(lambda p: _truncated_sum(p, 13, 128), id="truncated-13-bit-accumulator"),
    ],
)
def test_verifier_accepts_another_accumulation(accumulate) -> None:
    inputs = _inputs()
    assert _accepts(inputs, _finish(accumulate(_products(inputs)), inputs))


def _one_output_off(inputs) -> torch.Tensor:
    out = _finish(_products(inputs).sum(-1), inputs).float()
    out[1, 7] += 1.0 + 0.1 * out[1, 7].abs()
    return out.to(torch.bfloat16)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda i: _finish(_products(i, round_weight=False).sum(-1), i), id="weight-unrounded"
        ),
        pytest.param(lambda i: _finish(_products(i, signed=False).sum(-1), i), id="int4-unsigned"),
        pytest.param(
            lambda i: _finish(_products(i, group_shift=1).sum(-1), i), id="next-group-scale"
        ),
        pytest.param(
            lambda i: _finish(_products(i).sum(-1), i, channel=False), id="channel-scale-dropped"
        ),
        pytest.param(
            lambda i: _finish(_products(i).sum(-1), i, token=False), id="token-scale-dropped"
        ),
        pytest.param(_one_output_off, id="one-output-off"),
        pytest.param(lambda i: torch.zeros(M, N, dtype=torch.bfloat16), id="all-zero"),
    ],
)
def test_verifier_rejects_an_inexact_step(build) -> None:
    inputs = _inputs()
    assert not _accepts(inputs, build(inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [0, 1])
def test_verifier_matches_the_docstring_oracle_near_its_bound(seed: int) -> None:
    """Outputs moved by growing multiples of the per-element bound: both judges agree."""
    inputs = _inputs(seed)
    reference = _finish(_products(inputs).double().sum(-1), inputs).double()
    bound = 5e-5 * K + 2e-2 * reference.abs()
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(M, N, generator=generator).sign()
    for factor in (0.0, 0.1, 0.3, 0.6, 0.9, 1.2, 2.0):
        for fraction in (1 / (M * N), 0.1, 1.0):
            moved = torch.rand(M, N, generator=generator) < fraction
            got = (reference + moved * direction * factor * bound).to(torch.bfloat16)
            assert _accepts(inputs, got) == _docstring_oracle(inputs, got), (factor, fraction)
