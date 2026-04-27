from collections import defaultdict

import pytest
import torch

from tests.test_base import _check_result


def _under_repo_tests(item: pytest.Item) -> bool:
    path = str(item.path)
    return "tests/" in path and "benchmarks/tests/" not in path


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


NON_RUNTIME_OPS_TIER_FILES = {
    "tests/ops/test_elementwise_caching_autotune.py",
    "tests/ops/test_elementwise_compile.py",
    "tests/ops/test_elementwise_config_dtype.py",
}

TILELANG_019_SKIP_REASON = (
    "Temporarily skipped while tracking TileLang 0.1.9 migration failures (#1039)."
)

TILELANG_019_KNOWN_FAILING_PATH_SUFFIXES = (
    "tests/ops/test_batch_norm.py",
    "tests/ops/test_fft.py",
    "tests/ops/test_topk_selector.py",
    "tests/ops/attention/test_deepseek_nsa.py",
    "tests/ops/attention/test_gqa_decode.py",
    "tests/ops/attention/test_gqa_decode_paged.py",
    "tests/ops/attention/test_gqa_sliding_window.py",
    "tests/ops/attention/test_gqa_sliding_window_varlen.py",
    "tests/ops/attention/test_mha_decode_paged.py",
)

TILELANG_019_KNOWN_FAILING_NODEIDS = {
    "tests/ops/test_engram.py::test_engram_gate_conv_bwd[1-32-256-dtype0-False]",
    "tests/ops/test_engram.py::test_engram_gate_conv_bwd[1-32-256-dtype1-False]",
    "tests/ops/test_engram.py::test_engram_gate_conv_bwd[2-64-512-dtype2-False]",
    "tests/ops/test_engram.py::test_engram_gate_conv_bwd[2-16-256-dtype3-False]",
    "tests/ops/test_engram.py::test_engram_decode[1-512-256-12-4-3-dtype0-False]",
    "tests/ops/test_engram.py::test_engram_decode[1-512-256-12-4-3-dtype1-False]",
    "tests/ops/test_engram.py::test_engram_decode[4-1024-512-20-4-5-dtype2-False]",
    "tests/ops/test_engram.py::test_engram_decode[8-512-256-18-4-3-dtype3-False]",
    "tests/ops/test_engram.py::test_engram_decode_multi_step",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype0-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype1-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-64-2-64-64-32-dtype2-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype3-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype4-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[1-128-4-64-64-32-dtype5-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-8192-4-64-64-64-dtype6-False]",
    "tests/ops/test_deltanet_fwd.py::test_deltanet_fwd[2-16384-4-64-64-64-dtype7-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype0-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype1-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-64-2-64-64-32-dtype2-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype3-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype4-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[1-128-4-64-64-32-dtype5-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-8192-4-64-64-64-dtype6-False]",
    "tests/ops/test_gated_deltanet_fwd.py::test_gated_deltanet_fwd[2-16384-4-64-64-64-dtype7-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-64-2-64-64-64-dtype0-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-64-2-64-64-64-dtype1-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-64-2-64-64-64-dtype2-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[1-128-4-64-64-64-dtype3-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[1-128-4-64-64-64-dtype4-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[1-128-4-64-64-64-dtype5-False]",
    "tests/ops/test_gla_chunkwise_fwd.py::test_gla_fwd[2-256-4-64-64-64-dtype6-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype0-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype1-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[2-64-2-64-64-32-dtype2-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype0-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype1-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[2-64-2-64-64-32-dtype2-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype0-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype1-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[2-64-2-64-64-64-dtype2-False]",
    "tests/ops/attention/test_deepseek_nsa_topk.py::test_nsa_topk_varlen_op[5-1024-32-128-16-1-16-32-32-128-dtype0-accum_dtype0-False]",
    "tests/ops/attention/test_deepseek_nsa_topk.py::test_nsa_topk_varlen_op[3-512-32-128-16-1-16-32-32-128-dtype1-accum_dtype1-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[1-4-64-64-dtype1-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[1-4-64-64-dtype2-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-8-64-64-dtype5-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-8-64-64-dtype6-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[1-4-64-64-dtype1-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[1-4-64-64-dtype2-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-8-64-64-dtype5-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-8-64-64-dtype6-False]",
    "tests/ops/attention/test_deepseek_nsa_cmp.py::test_nsa_cmp_fwd_varlen_op[9-8192-32-128-128-16-0.08838834764831845-32-32-128-128-dtype0-accum_dtype0-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype3-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype4-False]",
    "tests/ops/test_deltanet_chunkwise_bwd.py::test_deltanet_bwd[1-128-4-64-64-32-dtype5-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[1-4-64-64-dtype0-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-4-128-128-dtype4-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode[2-8-64-64-dtype3-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[1-4-64-64-dtype0-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-4-128-128-dtype4-False]",
    "tests/ops/test_deltanet_recurrence.py::test_deltanet_decode_multi_step[2-8-64-64-dtype3-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype3-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype4-False]",
    "tests/ops/test_gated_deltanet_chunkwise_bwd.py::test_gated_deltanet_bwd[1-128-4-64-64-32-dtype5-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode[1-4-64-64-dtype0-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode[2-4-128-128-dtype4-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode[2-8-64-64-dtype3-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode_multi_step[1-4-64-64-dtype0-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode_multi_step[2-4-128-128-dtype4-False]",
    "tests/ops/test_gated_deltanet_recurrence.py::test_gated_deltanet_decode_multi_step[2-8-64-64-dtype3-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype3-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype4-False]",
    "tests/ops/test_gla_chunkwise_bwd.py::test_gla_bwd[1-128-4-64-64-64-dtype5-False]",
    "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[1-2-128-4-128-32-dtype3-False]",
    "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[1-2-64-4-64-32-dtype0-False]",
    "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[1-2-64-4-64-32-dtype1-False]",
    "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[2-2-64-4-64-32-dtype4-False]",
    "tests/ops/test_mamba.py::test_ssd_chunk_scan_fwd[2-4-64-8-64-64-dtype2-False]",
    "tests/ops/test_mamba.py::test_ssd_decode[1-4-64-16-1-dtype0-False]",
    "tests/ops/test_mamba.py::test_ssd_decode[1-4-64-16-1-dtype1-False]",
    "tests/ops/test_mamba.py::test_ssd_decode[2-8-128-64-4-dtype3-False]",
    "tests/ops/test_mamba.py::test_ssd_decode[2-8-64-32-2-dtype2-False]",
    "tests/ops/test_mhc_post.py::test_mhc_post_op[2-4-1920-dtype1-False]",
    "tests/ops/test_mhc_post.py::test_mhc_post_op[4-4-2560-dtype2-False]",
    "tests/ops/test_norm_ops.py::TestBatchNormCustomOp::test_fwd_torch_compile_smoke",
    "tests/ops/test_norm_ops.py::TestBatchNormCustomOp::test_bwd_torch_compile_smoke",
}

TILELANG_019_KNOWN_FAILING_PREFIXES = (
    "tests/test_autotune.py::test_mha_kernel_autotune",
    "tests/test_compile.py::test_mha_kernel_compile",
)

def _get_callspec_params(item: pytest.Item) -> dict | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None
    return getattr(callspec, "params", None)


def _freeze_value(value: object) -> object:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(val)) for key, val in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=str))
    return value


def _without_dtype(params: dict) -> tuple[tuple[str, object], ...]:
    return tuple(
        sorted((key, _freeze_value(value)) for key, value in params.items() if key != "dtype")
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate explicit test tier assignments."""
    tier_errors: list[str] = []
    tier_names = ("smoke", "full", "nightly")
    tilelang_019_skip = pytest.mark.skip(reason=TILELANG_019_SKIP_REASON)

    for item in items:
        path = str(item.path)
        if not _under_repo_tests(item):
            continue
        if (
            item.nodeid in TILELANG_019_KNOWN_FAILING_NODEIDS
            or any(path.endswith(suffix) for suffix in TILELANG_019_KNOWN_FAILING_PATH_SUFFIXES)
            or any(
                item.nodeid.startswith(prefix) for prefix in TILELANG_019_KNOWN_FAILING_PREFIXES
            )
        ):
            item.add_marker(tilelang_019_skip)

        tiers = [name for name in tier_names if item.get_closest_marker(name) is not None]
        if len(tiers) != 1:
            tier_errors.append(
                f"{item.nodeid}: expected exactly one tier marker, found {tiers or 'none'}"
            )

    ops_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    for item in items:
        path = str(item.path)
        if "tests/ops/" not in path or "benchmarks/tests/" in path:
            continue
        test_name = getattr(item, "originalname", item.name)
        ops_groups[(path, test_name)].append(item)

    for (_path, _test_name), group in ops_groups.items():
        if any(_path.endswith(path) for path in NON_RUNTIME_OPS_TIER_FILES):
            continue

        non_xfail_items = [
            item for item in group if item.get_closest_marker("xfail") is None
        ]
        smoke_items = [
            item for item in group if item.get_closest_marker("smoke") is not None
        ]

        # Smoke cases must never be xfail (checked before tune gate)
        for item in smoke_items:
            if item.get_closest_marker("xfail") is not None:
                tier_errors.append(f"{item.nodeid}: smoke cases must not be xfail")

        # For count and ordering checks, only consider non-xfail smoke cases
        valid_smoke_items = [
            item for item in smoke_items if item.get_closest_marker("xfail") is None
        ]

        if non_xfail_items:
            if len(valid_smoke_items) < 1:
                tier_errors.append(
                    f"{non_xfail_items[0].nodeid}: each test must have at least one smoke case"
                )
            else:
                # All smoke cases must appear as the first N non-xfail items
                expected_smoke = non_xfail_items[: len(valid_smoke_items)]
                if valid_smoke_items != expected_smoke:
                    tier_errors.append(
                        f"{non_xfail_items[0].nodeid}: all smoke cases must appear "
                        f"as the first {len(valid_smoke_items)} non-xfail cases of each test"
                    )

        dtype_supported: set[object] = set()
        dtype_smoke: set[object] = set()
        smoke_signatures: set[tuple[tuple[str, object], ...]] = set()
        dtype_cases_present = False

        for item in non_xfail_items:
            params = _get_callspec_params(item)
            if not params or "dtype" not in params:
                continue

            dtype_cases_present = True
            dtype_supported.add(params["dtype"])

            if item.get_closest_marker("smoke") is not None:
                dtype_smoke.add(params["dtype"])
                smoke_signatures.add(_without_dtype(params))

        if dtype_cases_present:
            missing_smoke_dtypes = dtype_supported - dtype_smoke
            if missing_smoke_dtypes:
                tier_errors.append(
                    f"{non_xfail_items[0].nodeid}: each dtype must have at least one smoke case; "
                    f"missing smoke for {sorted(str(dtype) for dtype in missing_smoke_dtypes)}"
                )

            for item in non_xfail_items:
                if item.get_closest_marker("full") is None:
                    continue

                params = _get_callspec_params(item)
                if not params or "dtype" not in params:
                    continue

                if _without_dtype(params) in smoke_signatures:
                    tier_errors.append(
                        f"{item.nodeid}: full cases must not differ from a smoke case only by dtype"
                    )

        first_tuned_item: pytest.Item | None = None
        full_tuned_items: list[pytest.Item] = []
        for item in group:
            params = _get_callspec_params(item)
            if params is None or "tune" not in params:
                continue

            tune = params["tune"]
            is_smoke = item.get_closest_marker("smoke") is not None
            if is_smoke and tune is True:
                tier_errors.append(f"{item.nodeid}: smoke cases must use tune=False")
            if tune is True:
                if first_tuned_item is None:
                    first_tuned_item = item
                if item.get_closest_marker("full") is not None:
                    full_tuned_items.append(item)
        if first_tuned_item is not None:
            if not full_tuned_items:
                tier_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be marked full"
                )
            elif len(full_tuned_items) > 1:
                tier_errors.append(
                    f"{group[0].path}::{group[0].originalname}: at most one tune=True case may be full"
                )
            elif full_tuned_items[0] is not first_tuned_item:
                tier_errors.append(
                    f"{first_tuned_item.nodeid}: the first tune=True case must be the only full tuned case"
                )

    if tier_errors:
        raise pytest.UsageError(
            "Invalid explicit test tier assignments detected:\n" + "\n".join(tier_errors)
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """After test execution, attach Op metadata from TestBase.check() to the item."""
    yield
    op_name = getattr(_check_result, "op_name", None)
    if op_name:
        item.user_properties.append(("op", op_name))
        op_module = getattr(_check_result, "op_module", None)
        if op_module:
            item.user_properties.append(("op_module", op_module))
        max_err = getattr(_check_result, "max_abs_err", None)
        if max_err is not None:
            item.user_properties.append(("max_abs_err", f"{max_err:.2e}"))
        _check_result.op_name = None
        _check_result.op_module = None
        _check_result.max_abs_err = None
