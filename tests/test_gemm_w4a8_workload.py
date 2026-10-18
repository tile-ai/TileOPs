"""Host-side W4A8 layout and numerical-contract tests.

Constructed outputs exercise accumulation choices and the output quality bounds without
running a kernel.
"""

import pytest
import torch

from workloads.gemm import (
    GemmW4A8Workload,
    gemm_w4a8_reference,
    repack_w4a8_weight,
    unpack_signed_int4,
)
from workloads.numerics import Request, compare_outputs, verify

M, N, K, GROUP = 4, 64, 2048, 128


@pytest.mark.smoke
def test_repack_w4a8_weight_places_each_nibble() -> None:
    """Every nibble lands where the layout formula puts it, encoded."""
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
def test_verifier_rejects_a_nonzero_output_in_a_zero_channel() -> None:
    """A zero channel scale gives an exactly zero output, inside the absolute bound."""
    inputs = list(_inputs())
    inputs[3][0] = 0
    workload = GemmW4A8Workload(M, N, K, GROUP)
    got = workload.ref_program(*inputs)
    got[0, 0] = 1e-3
    with pytest.raises(AssertionError, match="exactly zero output"):
        compare_outputs(got, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.smoke
def test_verifier_retains_the_complete_output_energy_bound() -> None:
    """Coherent contamination inside each element's bound still violates global quality."""
    inputs = list(_inputs())
    inputs[3].fill_(1)
    inputs[4].fill_(1)
    workload = GemmW4A8Workload(M, N, K, GROUP)
    reference = workload.ref_program(*inputs)
    contraction = _products(inputs).double().sum(-1)
    radius = 5e-5 * K + 2e-2 * contraction.abs()
    accumulator = (contraction + 0.5 * radius).float()
    assert ((accumulator.double() - contraction).abs() <= radius).all()
    got = ((accumulator * inputs[3][None, :]) * inputs[4][:, None]).to(torch.bfloat16)
    torch.testing.assert_close(got, reference, atol=5e-5 * K, rtol=2e-2)
    with pytest.raises(AssertionError, match="combined-energy accuracy"):
        compare_outputs(got, reference, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("index", [0, 2, 3, 4], ids=["activation", "group", "channel", "token"])
def test_verifier_requires_finite_numeric_inputs(index: int) -> None:
    inputs = list(_inputs())
    inputs[index] = inputs[index].clone()
    inputs[index].reshape(-1)[0] = float("nan")
    with pytest.raises(AssertionError, match="numerical contract requires finite"):
        GemmW4A8Workload(M, N, K, GROUP).verification(*inputs)


@pytest.mark.smoke
def test_verifier_retains_the_per_element_output_bound() -> None:
    inputs = list(_inputs())
    inputs[3].fill_(1.5)
    inputs[4].fill_(1.5)
    workload = GemmW4A8Workload(M, N, K, GROUP)
    reference = workload.ref_program(*inputs)
    contraction = _products(inputs).double().sum(-1)
    radius = 5e-5 * K + 2e-2 * contraction.abs()
    accumulator = contraction.float()
    accumulator[0, 0] = contraction[0, 0] + 0.9 * radius[0, 0]
    assert ((accumulator.double() - contraction).abs() <= radius).all()
    got = ((accumulator * inputs[3][None, :]) * inputs[4][:, None]).to(torch.bfloat16)
    difference = got.double() - reference.double()
    assert (
        difference.square().sum()
        <= 5e-5 * (got.double().square() + reference.double().square()).sum()
    )
    assert (difference.abs() > 5e-5 * K + 2e-2 * reference.double().abs()).any()
    with pytest.raises(AssertionError, match="absolute and relative W4A8"):
        compare_outputs(got, reference, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [64, 128])
def test_semantic_reference_binds_the_explicit_group_size(group_size: int) -> None:
    k, n = 256, 16
    activation = torch.full((1, k), 0.25).to(torch.float8_e4m3fn)
    packed = torch.full((n, k // 2), 0x44, dtype=torch.uint8)
    group = (
        (torch.arange(1, k // group_size + 1).float() / 64).expand(n, -1).to(torch.float8_e4m3fn)
    )
    result = gemm_w4a8_reference(
        activation, packed, group, torch.ones(n), torch.ones(1), group_size=group_size
    )
    expected = sum(group[0].float().tolist()) * group_size
    torch.testing.assert_close(result, torch.full_like(result, expected), atol=0, rtol=0)


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [0, -1])
def test_semantic_reference_refuses_a_nonpositive_group_size(group_size: int) -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="group_size must be positive"):
        gemm_w4a8_reference(*inputs, group_size=group_size)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kind",
    [
        "normal",
        "zero-weight",
        "zero-channel",
        "infinity",
        "nan",
    ],
)
def test_shared_verification_accepts_correct_degenerate_outputs(kind):
    m, n, k, group = 1, 16, 128, 128
    activation = torch.full((m, k), 0.25).to(torch.float8_e4m3fn)
    packed = torch.full((n, k // 2), 0x44, dtype=torch.uint8)
    scales = torch.full((n, k // group), 0.0625).to(torch.float8_e4m3fn)
    channel = torch.ones(n)
    token = torch.ones(m)
    if kind == "normal":
        scales = (0.03 + 0.05 * torch.rand_like(scales.float())).to(scales.dtype)
    if kind == "zero-weight":
        packed.zero_()
    if kind == "zero-channel":
        channel.zero_()
    if kind == "infinity":
        channel.fill_(torch.finfo(torch.float32).max)
    if kind == "nan":
        channel.zero_()
        token.fill_(torch.finfo(torch.float32).max)
    inputs = (activation, packed, scales, channel, token)
    w = GemmW4A8Workload(m, n, k, group)
    e = w.verification(*inputs)
    r = w.ref_program(*inputs)
    result = verify(
        w.ref_program,
        inputs,
        evidence=e,
        requests={"exact-semantic-output": Request(w.ref_program, inputs)},
    )
    assert result["exact-semantic-output"].checked_outputs == 1
    assert len(e.controls) == 1
    bad = e.controls[0].run(w.ref_program, inputs)
    assert bad.shape == r.shape and bad.dtype == r.dtype and bad.device == r.device
    torch.testing.assert_close(
        bad.reshape(-1)[1:], r.reshape(-1)[1:], atol=0, rtol=0, equal_nan=True
    )
    assert bool(bad.reshape(-1)[0].isnan()) != bool(r.reshape(-1)[0].isnan())
    with pytest.raises(AssertionError):
        compare_outputs(bad, r, e)
