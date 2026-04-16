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


_SMOKE_OP_NAME_PARAM_KEYS = ("op_kind", "op_name", "op_cls_path", "op_cls")

_SHAPE_PARAM_GROUPS = (
    ("a_shape", "b_shape"),
    ("shape",),
    ("m", "n"),
    ("M", "N"),
    ("M", "seq_len", "d"),
    ("batch", "seq", "hidden"),
    ("batch", "seq_len", "hidden"),
    ("b0", "b1", "b2", "n"),
    ("seq_len", "num_heads"),
    ("seq_len", "d_model"),
    ("batch", "heads", "dim_k", "dim_v"),
    ("batch", "heads", "heads_kv", "dim"),
    ("batch", "heads", "heads_kv", "seq_len_kv", "dim"),
    ("batch", "heads", "heads_kv", "seqlen_kv", "dim", "page_size"),
    ("batch", "heads", "seq_len_q", "seq_len_kv", "dim", "dim_tail", "topk"),
    ("batch", "heads", "seqlen_q", "seqlen_kv", "dim", "page_size"),
    ("batch", "seq_len", "heads", "heads_kv", "dim"),
    ("batch", "seq_len", "heads", "dim_k", "dim_v", "chunk_size"),
    ("batch", "heads", "seq_len", "dim_k", "dim_v", "chunk_size"),
    ("batch", "seq", "heads", "heads_kv", "dim", "wl", "wr"),
    ("batch", "seqlens_q", "seqlens_k", "heads", "heads_kv", "dim", "wl", "wr"),
    ("b", "h", "s_q", "s_kv", "d"),
    ("batch_size", "seq_len", "heads", "dim", "chunk_size"),
    ("seq_num", "c_seq_len", "heads", "dim", "group", "selected_block_num"),
    ("seq_num", "c_seq_len", "heads", "dim_k", "dim_v", "group"),
    ("batch", "heads", "c_seq_len", "dim", "block_size", "groups", "selected_blocks"),
    ("batch", "num_chunks", "chunk_len", "n_heads", "d_head", "d_state"),
    ("batch", "num_chunks", "chunk_len", "n_heads", "d_head", "d_state", "n_groups"),
    ("batch", "num_chunks", "n_heads", "d_state"),
    ("batch", "n_heads", "d_head", "d_state", "n_groups"),
    ("batch", "d_mem", "d", "max_conv_len", "conv_kernel_size", "dilation"),
    ("batch", "n_expand", "c_x"),
    ("N", "C", "spatial"),
    ("batch_sum", "batch_count", "N", "K"),
    ("num_tokens", "num_experts", "top_k", "hidden_size", "ffn_size"),
    ("total_tokens", "top_k", "num_experts", "hidden_size"),
    ("total_tokens", "top_k", "hidden_size"),
    ("batch", "seq_len_kv", "kv_group", "index_dim"),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk"),
    ("n_total",),
    ("n",),
)

_OP_NAME_SUFFIXES = (
    "_smoke_float16",
    "_smoke_bfloat16",
    "_smoke_float32",
    "_smoke_int32",
    "_smoke_int64",
    "_smoke_bool",
    "_signed_zero_with_nan",
    "_nan_propagation",
    "_optimized_large",
    "_non_contiguous",
    "_non_contig",
    "_same_shape",
    "_dtype_size",
    "_min_gt_max",
    "_upper_only",
    "_lower_only",
    "_all_true",
    "_all_false",
    "_edge_cases",
    "_edge_case",
    "_spec_keepdim",
    "_spec_basic",
    "_spec_dim0_keepdim",
    "_spec_dim1_3d",
    "_spec_dim0",
    "_spec_dim",
    "_spec_1d",
    "_dim0_keepdim",
    "_3d_dim0_keepdim",
    "_4d_dim0_keepdim",
    "_3d_dim0",
    "_4d_dim0",
    "_keepdim",
    "_broadcast",
    "_strategies",
    "_signed_zero",
    "_tiled",
    "_3d",
    "_4d",
    "_1d",
    "_dim",
    "_edge",
    "_op",
)


def _normalize_dtype(dtype: object) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).split(".")[-1]
    return str(dtype)


def _normalize_shape_value(value: object) -> tuple:
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


_NON_SHAPE_PARAM_NAMES = {
    "dtype",
    "in_dtype",
    "out_dtype",
    "accum_dtype",
    "compute_dtype",
    "weight_dtype",
    "input_dtype",
    "in_dtype_str",
    "out_dtype_str",
    "tune",
    "training",
    "causal",
    "is_causal",
    "renormalize",
    "with_correction_bias",
    "transpose_a",
    "transpose_b",
    "kernel_name",
    "extra_kwargs",
    "op_cls",
    "op_kind",
    "op_name",
    "op_cls_path",
    "scoring_func",
    "sm_scale",
    "scale",
    "q_start_index_s",
    "offsets",
}


def _is_shape_like_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, torch.Size):
        return True
    if isinstance(value, (tuple, list)):
        return all(isinstance(v, int) for v in value)
    return False


def _fallback_shape_signature(params: dict) -> tuple | None:
    shape_items: list[tuple[str, tuple]] = []
    for key, value in params.items():
        if key in _NON_SHAPE_PARAM_NAMES:
            continue
        if not _is_shape_like_value(value):
            continue
        shape_items.append((key, _normalize_shape_value(value)))

    if not shape_items:
        return None

    return tuple(shape_items)


def _extract_shape_signature(item: pytest.Item) -> tuple | None:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return None

    params = callspec.params
    for group in _SHAPE_PARAM_GROUPS:
        if all(name in params for name in group):
            if group == ("a_shape", "b_shape"):
                return (
                    _normalize_shape_value(params["a_shape"]),
                    _normalize_shape_value(params["b_shape"]),
                )
            if group == ("shape",):
                return (_normalize_shape_value(params["shape"]),)
            return (tuple(params[name] for name in group),)
    return _fallback_shape_signature(params)


def _infer_op_key(item: pytest.Item) -> str:
    callspec = getattr(item, "callspec", None)
    params = callspec.params if callspec is not None else {}

    for key in _SMOKE_OP_NAME_PARAM_KEYS:
        if key not in params:
            continue
        value = params[key]
        if key == "op_cls_path":
            return str(value).split(".")[-1]
        if key == "op_cls":
            return getattr(value, "__name__", str(value))
        return str(value)

    name = getattr(item, "originalname", item.name)
    if name.startswith("test_"):
        name = name[5:]

    changed = True
    while changed:
        changed = False
        for suffix in _OP_NAME_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    return name


def _format_values(values: set[str]) -> str:
    return ", ".join(sorted(values))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate explicit test tier assignments."""
    tier_errors: list[str] = []
    tier_names = ("smoke", "full", "nightly")

    for item in items:
        if not _under_repo_tests(item):
            continue
        tiers = [name for name in tier_names if item.get_closest_marker(name) is not None]
        if len(tiers) != 1:
            tier_errors.append(
                f"{item.nodeid}: expected exactly one tier marker, found {tiers or 'none'}"
            )

    ops_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    smoke_contract_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    for item in items:
        path = str(item.path)
        if "tests/ops/" not in path or "benchmarks/tests/" in path:
            continue
        test_name = getattr(item, "originalname", item.name)
        ops_groups[(path, test_name)].append(item)
        if any(path.endswith(skip_path) for skip_path in NON_RUNTIME_OPS_TIER_FILES):
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is None or "dtype" not in callspec.params:
            continue
        if _extract_shape_signature(item) is None:
            continue
        smoke_contract_groups[(path, _infer_op_key(item))].append(item)

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

        if valid_smoke_items:
            # All smoke cases must appear as the first N non-xfail items
            expected_smoke = non_xfail_items[: len(valid_smoke_items)]
            if valid_smoke_items != expected_smoke:
                tier_errors.append(
                    f"{non_xfail_items[0].nodeid}: all smoke cases must appear "
                    f"as the first {len(valid_smoke_items)} non-xfail cases of each test"
                )

        smoke_signatures: set[tuple[tuple[str, object], ...]] = set()

        for item in non_xfail_items:
            params = _get_callspec_params(item)
            if not params or "dtype" not in params:
                continue

            if item.get_closest_marker("smoke") is not None:
                smoke_signatures.add(_without_dtype(params))

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

    for (path, op_key), group in smoke_contract_groups.items():
        non_xfail_items = [item for item in group if item.get_closest_marker("xfail") is None]
        if not non_xfail_items:
            continue

        supported_dtypes = {
            _normalize_dtype(item.callspec.params["dtype"]) for item in non_xfail_items
        }
        valid_smoke_items = [
            item
            for item in non_xfail_items
            if item.get_closest_marker("smoke") is not None
        ]
        smoke_dtypes = {
            _normalize_dtype(item.callspec.params["dtype"]) for item in valid_smoke_items
        }
        smoke_shapes = {
            _extract_shape_signature(item) for item in valid_smoke_items
        }

        if len(valid_smoke_items) != len(supported_dtypes):
            tier_errors.append(
                f"{path}::{op_key}: expected exactly {len(supported_dtypes)} smoke cases "
                f"(one typical shape x all supported dtypes), found {len(valid_smoke_items)}"
            )

        if smoke_dtypes != supported_dtypes:
            missing = supported_dtypes - smoke_dtypes
            extra = smoke_dtypes - supported_dtypes
            details: list[str] = []
            if missing:
                details.append(f"missing [{_format_values(missing)}]")
            if extra:
                details.append(f"unexpected [{_format_values(extra)}]")
            tier_errors.append(
                f"{path}::{op_key}: smoke dtype coverage must match supported dtypes; "
                + ", ".join(details)
            )

        if len(smoke_shapes) > 1:
            shape_desc = ", ".join(str(shape) for shape in sorted(smoke_shapes, key=str))
            tier_errors.append(
                f"{path}::{op_key}: smoke cases must use exactly one typical shape, found {shape_desc}"
            )

    if tier_errors:
        raise pytest.UsageError(
            "Invalid explicit test tier assignments detected:\n" + "\n".join(tier_errors)
        )

    # Runtime policy: for tests/ops, only smoke-tier cases are allowed to run.
    # Non-smoke cases are always deselected, regardless of -m selection.
    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []
    for item in items:
        path = str(item.path)
        if "tests/ops/" in path and item.get_closest_marker("smoke") is None:
            deselected.append(item)
            continue
        kept.append(item)

    if deselected and items:
        items[0].config.hook.pytest_deselected(items=deselected)
    items[:] = kept


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
