"""Host-side W4A8 layout and numerical-contract tests.

Constructed outputs exercise finite accumulator choices, exact scaling associations and
output quality bounds without running a kernel.
"""

import math
import struct

import pytest
import torch

from workloads.gemm import (
    GemmW4A8Workload,
    _w4a8_accumulator_contains,
    _w4a8_epilogue,
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
    """A zero channel scale must survive every allowed accumulation and scale association."""
    inputs = list(_inputs())
    inputs[3][0] = 0
    workload = GemmW4A8Workload(M, N, K, GROUP)
    got = workload.ref_program(*inputs)
    got[0, 0] = 1e-3
    with pytest.raises(AssertionError, match="exact BF16 rounding"):
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


def _scalar_epilogue(acc: float, channel: float, token: float, order: str) -> int:
    """Independent scalar IEEE binary32 multiplications and BF16 round-to-nearest-even."""

    def f32(value):
        try:
            return struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError:
            return math.copysign(math.inf, value)

    if order == "channel-first":
        result = f32(f32(acc * channel) * token)
    elif order == "token-first":
        result = f32(f32(acc * token) * channel)
    else:
        result = f32(acc * f32(channel * token))
    bits = struct.unpack("I", struct.pack("f", result))[0]
    if math.isnan(result):
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


@pytest.mark.smoke
@pytest.mark.parametrize(
    "center, channel, token, order",
    [
        pytest.param(1.0, 0.75, 1.25, "channel-first", id="positive-scales"),
        pytest.param(-1.0, -0.75, 1.25, "token-first", id="negative-scale"),
        pytest.param(0.0, 0.0, 1.0, "scale-first", id="zero-scale"),
        pytest.param(
            2.0**-126, 2.0**-126, 2.0**126, "channel-first", id="underflow-before-rescale"
        ),
        pytest.param(
            1.0, torch.finfo(torch.float32).max, 0.0, "token-first", id="overflow-before-zero"
        ),
        pytest.param(
            0.0, torch.finfo(torch.float32).max, 2.0, "scale-first", id="scale-product-overflow"
        ),
    ],
)
def test_accumulator_membership_matches_exhaustive_scalar_image(
    center: float, channel: float, token: float, order: str
) -> None:
    """All 65,536 BF16 codes agree with exhaustive forward images of five FP32 values."""
    middle = torch.tensor(center, dtype=torch.float32)
    values = [middle]
    for direction in (-float("inf"), float("inf")):
        current = middle
        for _ in range(2):
            current = torch.nextafter(current, torch.tensor(direction))
            values.append(current)
    scalars = sorted(set(value.item() for value in values))
    codes = {_scalar_epilogue(value, channel, token, order) for value in scalars}
    candidates = torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    expected = torch.zeros_like(candidates, dtype=torch.bool)
    bits = candidates.view(torch.int16).to(torch.int32) & 0xFFFF
    for code in codes:
        expected |= candidates.isnan() if code == 0x7FC0 else bits == code
    if 0 in codes or 0x8000 in codes:
        expected |= candidates == 0
    actual = _w4a8_accumulator_contains(
        candidates,
        torch.tensor(scalars[0]),
        torch.tensor(scalars[-1]),
        torch.tensor(channel),
        torch.tensor(token),
        order,
    )
    assert torch.equal(actual, expected)


@pytest.mark.smoke
@pytest.mark.parametrize("order", ["channel-first", "token-first", "scale-first"])
def test_exact_epilogue_matches_the_independent_scalar_rounding(order: str) -> None:
    """Each allowed association has its own exact finite and nonfinite forward oracle."""
    cases = [(1.0, 0.75, 1.25), (1.0, 3.4e38, 0.0), (1e-38, 1e-38, 1e38)]
    for acc, channel, token in cases:
        operands = tuple(
            torch.tensor(value, dtype=torch.float32) for value in (acc, channel, token)
        )
        actual = _w4a8_epilogue(*operands, order)
        expected = _scalar_epilogue(*(value.item() for value in operands), order)
        assert (
            actual.isnan()
            if expected == 0x7FC0
            else (actual.view(torch.int16).item() & 0xFFFF) == expected
        )


@pytest.mark.smoke
@pytest.mark.parametrize("index", [0, 2, 3, 4], ids=["activation", "group", "channel", "token"])
def test_verifier_requires_finite_numeric_inputs(index: int) -> None:
    inputs = list(_inputs())
    inputs[index] = inputs[index].clone()
    inputs[index].reshape(-1)[0] = float("nan")
    with pytest.raises(AssertionError, match="numerical contract requires finite"):
        GemmW4A8Workload(M, N, K, GROUP).verification(*inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("channel", [0.0, 1.0], ids=["constant-image", "monotone-image"])
def test_accumulator_membership_rejects_an_empty_interval(channel: float) -> None:
    assert not _w4a8_accumulator_contains(
        torch.tensor(0.0, dtype=torch.bfloat16),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(channel),
        torch.tensor(1.0),
        "channel-first",
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "order, value, channel, token",
    [
        pytest.param(
            "channel-first",
            8.028412818908691,
            1.041380524635315,
            0.5737454295158386,
            id="channel-first",
        ),
        pytest.param(
            "token-first",
            8.028919219970703,
            0.8352468609809875,
            1.1393494606018066,
            id="token-first",
        ),
        pytest.param(
            "scale-first",
            8.05581283569336,
            0.9909803867340088,
            0.5186704397201538,
            id="scale-first",
        ),
    ],
)
def test_verifier_accepts_a_nonreference_accumulator_with_each_scale_association(
    order: str, value: float, channel: float, token: float
) -> None:
    """Each association has a nonreference legal result distinguishable from another association."""
    m, n, k, group = 1, 16, 128, 128
    activation = torch.full((m, k), 0.25).to(torch.float8_e4m3fn)
    packed = torch.full((n, k // 2), 0x44, dtype=torch.uint8)
    scale = torch.full((n, k // group), 0.0625).to(torch.float8_e4m3fn)
    channels = torch.ones(n)
    channels[0] = channel
    tokens = torch.full((m,), token)
    inputs = (activation, packed, scale, channels, tokens)
    workload = GemmW4A8Workload(m, n, k, group)
    reference = workload.ref_program(*inputs)
    contraction = torch.full((m, n), k * 0.25 * 0.25, dtype=torch.float64)
    accumulator = contraction.float()
    accumulator[0, 0] = value
    radius = 5e-5 * k + 2e-2 * contraction.abs()
    assert ((accumulator.double() - contraction).abs() <= radius).all()
    result = _w4a8_epilogue(accumulator, channels[None, :], tokens[:, None], order)
    assert not torch.equal(result, reference)
    codes = {
        _scalar_epilogue(value, channel, token, mode)
        for mode in ("channel-first", "token-first", "scale-first")
    }
    assert len(codes) > 1
    assert (result[0, 0].view(torch.int16).item() & 0xFFFF) == _scalar_epilogue(
        value, channel, token, order
    )
    torch.testing.assert_close(result, reference, atol=5e-5 * k, rtol=2e-2)
    difference = result.double() - reference.double()
    assert (
        difference.square().sum()
        <= 5e-5 * (result.double().square() + reference.double().square()).sum()
    )
    compare_outputs(result, reference, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("side", [-1, 1], ids=["below-lower", "above-upper"])
def test_verifier_refuses_an_accumulator_outside_its_input_bound(side: int) -> None:
    inputs = list(_inputs())
    inputs[3].fill_(0.5)
    inputs[4].fill_(0.5)
    workload = GemmW4A8Workload(M, N, K, GROUP)
    reference = workload.ref_program(*inputs)
    contraction = _products(inputs).double().sum(-1)
    radius = 5e-5 * K + 2e-2 * contraction.abs()
    accumulator = contraction.float()
    accumulator[0, 0] = contraction[0, 0] + side * 2 * radius[0, 0]
    got = ((accumulator * inputs[3][None, :]) * inputs[4][:, None]).to(torch.bfloat16)
    torch.testing.assert_close(got, reference, atol=5e-5 * K, rtol=2e-2)
    difference = got.double() - reference.double()
    assert (
        difference.square().sum()
        <= 5e-5 * (got.double().square() + reference.double().square()).sum()
    )
    low = ((contraction - radius).float() * 0.5 * 0.5).to(torch.bfloat16)
    high = ((contraction + radius).float() * 0.5 * 0.5).to(torch.bfloat16)
    assert got[0, 0] < low[0, 0] if side < 0 else got[0, 0] > high[0, 0]
    with pytest.raises(AssertionError, match="exact BF16 rounding"):
        compare_outputs(got, reference, workload.verification(*inputs))


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


def _overflow_inputs(kind):
    activation = torch.ones((1, 128)).to(torch.float8_e4m3fn)
    packed = torch.full((16, 64), 0x11, dtype=torch.uint8)
    group = torch.ones((16, 1)).to(torch.float8_e4m3fn)
    maximum = torch.finfo(torch.float32).max
    channel = torch.full((16,), maximum if kind == "infinity" else 0, dtype=torch.float32)
    token = torch.full((1,), 1.0 if kind == "infinity" else maximum)
    return activation, packed, group, channel, token


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["infinity", "nan"])
def test_verifier_accepts_the_reference_nonfinite_class(kind: str) -> None:
    """Finite inputs may overflow exact scaling into the reference's output class."""
    inputs = _overflow_inputs(kind)
    workload = GemmW4A8Workload(1, 16, 128, 128)
    reference = workload.ref_program(*inputs)
    accumulator = torch.full((1, 16), 128.0)
    result = _w4a8_epilogue(accumulator, inputs[3][None, :], inputs[4][:, None], "token-first")
    if kind == "infinity":
        assert reference.isposinf().all()
    else:
        assert reference.isnan().all()
    compare_outputs(result, reference, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["infinity", "nan"])
def test_verifier_refuses_a_changed_nonfinite_class(kind: str) -> None:
    inputs = list(_overflow_inputs(kind))
    if kind == "infinity":
        inputs[3].fill_(1e-38)
        inputs[4].fill_(torch.finfo(torch.float32).max)
    workload = GemmW4A8Workload(1, 16, 128, 128)
    reference = workload.ref_program(*inputs)
    accumulator = torch.full((1, 16), 128.0)
    assert torch.equal(_products(inputs).double().sum(-1), accumulator.double())
    alternative = _w4a8_epilogue(
        accumulator, inputs[3][None, :], inputs[4][:, None], "channel-first"
    )
    assert alternative.isfinite().all()
    assert (
        _w4a8_epilogue(accumulator, inputs[3][None, :], inputs[4][:, None], "token-first")
        .isfinite()
        .equal(reference.isfinite())
    )
    result = reference.clone()
    result[0, 0] = alternative[0, 0]
    assert result[0, 0] == alternative[0, 0]
    assert torch.equal(result[:, 1:].isfinite(), reference[:, 1:].isfinite())
    with pytest.raises(AssertionError, match="absolute and relative W4A8"):
        compare_outputs(result, reference, workload.verification(*inputs))


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [0, -1])
def test_semantic_reference_refuses_a_nonpositive_group_size(group_size: int) -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="group_size must be positive"):
        gemm_w4a8_reference(*inputs, group_size=group_size)


@pytest.mark.smoke
def test_full_verifier_matches_an_independent_small_output_domain() -> None:
    """All pairs in a finite BF16 output domain agree with an enumerated accumulator oracle."""
    activation = torch.full((1, 128), 0.25).to(torch.float8_e4m3fn)
    packed = torch.full((16, 64), 0x44, dtype=torch.uint8)
    group = torch.full((16, 1), 0.0625).to(torch.float8_e4m3fn)
    inputs = (activation, packed, group, torch.ones(16), torch.ones(1))
    workload = GemmW4A8Workload(1, 16, 128, 128)
    reference = workload.ref_program(*inputs)
    assert torch.equal(reference, torch.full_like(reference, 8))
    radius = 5e-5 * 128 + 2e-2 * 8
    low = struct.unpack("<I", struct.pack("<f", 8 - radius))[0] - 1
    high = struct.unpack("<I", struct.pack("<f", 8 + radius))[0] + 1
    image = set()
    for bits in range(low, high + 1):
        accumulator = struct.unpack("<f", struct.pack("<I", bits))[0]
        if abs(accumulator - 8) <= radius:
            image.add(_scalar_epilogue(accumulator, 1.0, 1.0, "token-first"))
    candidates = (7.5, 7.875, 8.0, 8.125, 8.5)
    for left in candidates:
        for right in candidates:
            values = (left, right)
            codes = [struct.unpack("<I", struct.pack("<f", item))[0] >> 16 for item in values]
            allowed = all(code in image for code in codes)
            allowed &= all(abs(item - 8) <= radius for item in values)
            error = sum((item - 8) ** 2 for item in values)
            energy = 30 * 8**2 + sum(item**2 for item in values)
            allowed &= error <= 5e-5 * energy
            result = reference.clone()
            result[0, :2] = torch.tensor(values)
            try:
                compare_outputs(result, reference, workload.verification(*inputs))
            except AssertionError:
                accepted = False
            else:
                accepted = True
            assert accepted == allowed, values


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
