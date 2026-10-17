"""CPU validation of W4A8 builder dimensions, integer domains and shared-memory limits."""

import itertools

import pytest

from tileops.kernels.gemm.w4a8_decode import _check_config as check_decode
from tileops.kernels.gemm.w4a8_decode import _shared_bytes
from tileops.kernels.gemm.w4a8_prefill import _check_config as check_prefill

pytestmark = pytest.mark.smoke


def _decode_config(**changes):
    return (
        dict(
            block_tokens=8,
            warpgroups=1,
            unit_warpgroups=1,
            block_k=128,
            num_stages=2,
            num_ctas=1,
        )
        | changes
    )


def _prefill_config(**changes):
    return (
        dict(
            block_m=64,
            num_stages=2,
            epilogue_tokens=64,
            epilogue_buffers=1,
            group_m=1,
            cluster=1,
            stream_k=0,
        )
        | changes
    )


def test_decode_validator_covers_the_finite_configuration_product() -> None:
    """Every token extent, warpgroup split, K block, legal stage and resident CTA count is checked."""
    budget, sms, k = 227 * 1024, 132, 1024
    checked = 0
    for nt, nw, uw, bk, n in itertools.product(
        (8, 16, 32), range(1, 5), (1, 2), (128, 256, 512), (16, 128)
    ):
        structural = nw % uw == 0 and not (uw == 2 and n % 128)
        base = _shared_bytes(nt, nw, uw, bk, 2)[0]
        stride = _shared_bytes(nt, nw, uw, bk, 3)[0] - base
        last_stage = 2 + max(0, (budget - base) // stride) if structural else 2
        tiles = -(-n // (64 * uw))
        items = tiles * -(-k // bk)
        for stages, ctas, m in itertools.product(
            range(2, last_stage + 2), range(1, min(sms, items) + 2), range(1, 33)
        ):
            config = _decode_config(
                block_tokens=nt,
                warpgroups=nw,
                unit_warpgroups=uw,
                block_k=bk,
                num_stages=stages,
                num_ctas=ctas,
            )
            expected = (
                structural
                and nt >= m
                and _shared_bytes(nt, nw, uw, bk, stages)[0] <= budget
                and ctas <= min(sms, items)
            )
            assert (check_decode(m, n, k, config, budget, sms) is None) == expected, (
                m,
                n,
                k,
                config,
            )
            checked += 1
    assert checked > 0


def test_prefill_validator_covers_the_finite_configuration_product() -> None:
    """Every tile, epilogue divisor, buffer count, split mode and cluster interaction is checked."""
    for bm, epilogue, buffers, split, cluster, n in itertools.product(
        (32, 48, 64, 80, 96, 128, 192, 256),
        (32, 48, 64, 80, 96, 128),
        (1, 2),
        (0, 1, 2),
        (1, 2),
        (128, 256),
    ):
        config = _prefill_config(
            block_m=bm,
            epilogue_tokens=epilogue,
            epilogue_buffers=buffers,
            stream_k=split,
            cluster=cluster,
        )
        expected = bm % epilogue == 0 and (cluster == 1 or (bm == 256 and n % 256 == 0))
        stages = 2
        while True:
            config["num_stages"] = stages
            try:
                check_prefill(n, config, 227 * 1024)
            except ValueError as error:
                if expected:
                    assert "shared memory" in str(error), (n, config, str(error))
                break
            assert expected, (n, config)
            stages += 1


@pytest.mark.parametrize(
    "config, match",
    [
        pytest.param(_decode_config(block_tokens=7), "block_tokens", id="token-below-domain"),
        pytest.param(_decode_config(block_tokens=33), "block_tokens", id="token-above-domain"),
        pytest.param(_decode_config(warpgroups=0), "warpgroups", id="no-consumer"),
        pytest.param(_decode_config(warpgroups=5), "warpgroups", id="too-many-consumers"),
        pytest.param(_decode_config(unit_warpgroups=3), "unit_warpgroups", id="invalid-unit-split"),
        pytest.param(_decode_config(block_k=127), "block_k", id="k-below-domain"),
        pytest.param(_decode_config(block_k=513), "block_k", id="k-above-domain"),
        pytest.param(_decode_config(num_stages=1), "num_stages", id="stage-lower-neighbor"),
        pytest.param(_decode_config(num_ctas=0), "num_ctas", id="cta-lower-neighbor"),
        pytest.param(_decode_config(num_ctas=17), "num_ctas", id="cta-upper-neighbor"),
        pytest.param(_decode_config(num_stages=2.0), "integer", id="fractional-type"),
        pytest.param(_decode_config(warpgroups=True), "integer", id="boolean-type"),
    ],
)
def test_decode_validator_refuses_one_invalid_dimension(config, match) -> None:
    reason = check_decode(1, 128, 1024, config, 227 * 1024, 132)
    assert reason is not None and match in reason


@pytest.mark.parametrize(
    "config, match",
    [
        pytest.param(_prefill_config(block_m=31), "block_m", id="tile-lower-neighbor"),
        pytest.param(_prefill_config(block_m=257), "block_m", id="tile-upper-neighbor"),
        pytest.param(_prefill_config(epilogue_tokens=31), "epilogue_tokens", id="invalid-epilogue"),
        pytest.param(_prefill_config(epilogue_tokens=48), "divide", id="nondividing-epilogue"),
        pytest.param(_prefill_config(epilogue_buffers=0), "buffers", id="buffer-lower-neighbor"),
        pytest.param(_prefill_config(epilogue_buffers=3), "buffers", id="buffer-upper-neighbor"),
        pytest.param(_prefill_config(stream_k=-1), "stream_k", id="split-lower-neighbor"),
        pytest.param(_prefill_config(stream_k=3), "stream_k", id="split-upper-neighbor"),
        pytest.param(_prefill_config(num_stages=1), "num_stages", id="stage-lower-neighbor"),
        pytest.param(_prefill_config(group_m=0), "group_m", id="group-lower-neighbor"),
        pytest.param(_prefill_config(cluster=0), "cluster", id="cluster-lower-neighbor"),
        pytest.param(_prefill_config(cluster=3), "cluster", id="cluster-upper-neighbor"),
        pytest.param(_prefill_config(cluster=2), "cluster", id="cluster-tile-interaction"),
        pytest.param(_prefill_config(num_stages=2.0), "integer", id="fractional-type"),
        pytest.param(_prefill_config(cluster=True), "integer", id="boolean-type"),
    ],
)
def test_prefill_validator_refuses_one_invalid_dimension(config, match) -> None:
    with pytest.raises(ValueError, match=match):
        check_prefill(128, config, 227 * 1024)


def test_decode_validator_checks_shared_memory_at_its_exact_boundary() -> None:
    config = _decode_config()
    required = _shared_bytes(8, 1, 1, 128, 2)[0]
    assert check_decode(1, 128, 1024, config, required, 132) is None
    assert "shared memory" in check_decode(1, 128, 1024, config, required - 1, 132)


def test_prefill_validator_checks_shared_memory_at_its_exact_boundary() -> None:
    config = _prefill_config()
    low, high = 0, 227 * 1024
    while low < high:
        middle = (low + high) // 2
        try:
            check_prefill(128, config, middle)
        except ValueError:
            low = middle + 1
        else:
            high = middle
    check_prefill(128, config, low)
    with pytest.raises(ValueError, match="shared memory"):
        check_prefill(128, config, low - 1)


@pytest.mark.parametrize("n, cluster", [(128, 1), (256, 1), (256, 2)])
def test_prefill_validator_bounds_group_ordering_by_the_index_product(n, cluster) -> None:
    config = _prefill_config(block_m=256, cluster=cluster)
    n_pairs = -(-n // 128) // cluster
    maximum = (2**31 - 1) // n_pairs
    for group in (1, 2, 3, 4, maximum):
        check_prefill(n, config | {"group_m": group}, 227 * 1024)
    with pytest.raises(ValueError, match="signed 32-bit tile indices"):
        check_prefill(n, config | {"group_m": maximum + 1}, 227 * 1024)


@pytest.mark.parametrize("n", [0, 15, 17])
def test_prefill_validator_requires_whole_positive_weight_units(n) -> None:
    with pytest.raises(ValueError, match="positive multiple"):
        check_prefill(n, _prefill_config(), 227 * 1024)


def test_layout_reference_extents_match_the_kernel_abi() -> None:
    import torch

    from tileops.kernels.gemm.w4a8_repack import GROUP_SIZE, W4A8_LAYOUT
    from tileops.utils import WARP_LANES
    from workloads.gemm import _W4A8_PACKING

    assert _W4A8_PACKING.bits == torch.iinfo(torch.uint8).bits // 2
    assert _W4A8_PACKING.unit_rows == W4A8_LAYOUT.unit_rows
    assert _W4A8_PACKING.block_rows == W4A8_LAYOUT.tile_rows
    assert _W4A8_PACKING.group_k == W4A8_LAYOUT.tile_k == GROUP_SIZE
    assert _W4A8_PACKING.warp_lanes == WARP_LANES
