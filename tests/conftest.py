import ast
from collections import defaultdict
from pathlib import Path

import pytest
import torch

from tests.test_base import _check_result

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_INIT_PATH = REPO_ROOT / "tileops" / "ops" / "__init__.py"
TIER_NAMES = ("smoke", "full", "nightly")
OPS_TEST_PREFIX = "tests/ops/"
AUTOTUNE_PARAM_NAMES = ("tune", "autotune")


# Canonical smoke ownership for tests/ops is centralized here.
# The collected smoke set is assigned dynamically during collection, so
# tests/ops sources do not need to preserve static pytest.mark.smoke markers.
# When a new public Op is exported from tileops/ops/__init__.py, add exactly
# one canonical smoke target here and let collection normalize the remaining
# tests/ops cases to full.
PUBLIC_OP_SMOKE_TARGETS: dict[str, tuple[str, str]] = {
    "AdaLayerNormOp": ("tests/ops/test_ada_layer_norm.py", "test_ada_layer_norm_op"),
    "AdaLayerNormZeroOp": (
        "tests/ops/test_ada_layer_norm_zero.py",
        "test_ada_layer_norm_zero_op",
    ),
    "AllOp": ("tests/ops/test_logical_reduce.py", "test_all_op"),
    "AmaxOp": ("tests/ops/test_reduce.py", "test_amax_op"),
    "AminOp": ("tests/ops/test_reduce.py", "test_amin_op"),
    "AnyOp": ("tests/ops/test_logical_reduce.py", "test_any_op"),
    "ArgmaxOp": ("tests/ops/test_argreduce.py", "test_argmax_op"),
    "ArgminOp": ("tests/ops/test_argreduce.py", "test_argmin_op"),
    "BatchNormBwdOp": ("tests/ops/test_batch_norm.py", "test_batch_norm_bwd"),
    "BatchNormFwdOp": ("tests/ops/test_batch_norm.py", "test_batch_norm_fwd"),
    "BinaryOp": ("tests/ops/test_binary_arith.py", "test_add_same_shape"),
    "Conv1dOp": ("tests/ops/test_conv1d.py", "test_conv1d"),
    "Conv2dOp": ("tests/ops/test_conv2d.py", "test_conv2d"),
    "Conv3dOp": ("tests/ops/test_conv3d.py", "test_conv3d"),
    "CountNonzeroOp": ("tests/ops/test_logical_reduce.py", "test_count_nonzero_op"),
    "CumprodOp": ("tests/ops/test_cumulative.py", "test_cumprod_op"),
    "CumsumOp": ("tests/ops/test_cumulative.py", "test_cumsum_op"),
    "DeepSeekSparseAttentionDecodeWithKVCacheOp": (
        "tests/ops/test_deepseek_dsa_decode.py",
        "test_sparse_mla_decode",
    ),
    "DeltaNetBwdOp": ("tests/ops/test_deltanet_chunkwise_bwd.py", "test_deltanet_bwd"),
    "DeltaNetDecodeOp": ("tests/ops/test_deltanet_recurrence.py", "test_deltanet_decode"),
    "DeltaNetFwdOp": ("tests/ops/test_deltanet_chunkwise_fwd.py", "test_deltanet_fwd"),
    "DeltaNetOp": ("tests/ops/test_deltanet_chunkwise_fwd.py", "test_deltanet_fwd"),
    "DropoutOp": ("tests/ops/test_dropout.py", "test_dropout_statistical_rate"),
    "FFTC2COp": ("tests/ops/test_fft.py", "test_fft_c2c"),
    "Fp8LightingIndexerOp": ("tests/ops/test_fp8_lighting_indexer.py", "test_indexer"),
    "Fp8QuantOp": ("tests/ops/test_fp8_quant.py", "test_fp8_quant_op"),
    "FusedAddLayerNormOp": (
        "tests/ops/test_fused_add_layer_norm.py",
        "test_fused_add_layer_norm_op",
    ),
    "FusedAddRmsNormOp": (
        "tests/ops/test_fused_add_rmsnorm.py",
        "test_fused_add_rmsnorm_op",
    ),
    "FusedGatedOp": ("tests/ops/test_fused_gated.py", "test_silu_and_mul_op"),
    "GatedDeltaNetBwdOp": (
        "tests/ops/test_gated_deltanet_chunkwise_bwd.py",
        "test_gated_deltanet_bwd",
    ),
    "GatedDeltaNetDecodeOp": (
        "tests/ops/test_gated_deltanet_recurrence.py",
        "test_gated_deltanet_decode",
    ),
    "GatedDeltaNetFwdOp": (
        "tests/ops/test_gated_deltanet_chunkwise_fwd.py",
        "test_gated_deltanet_fwd",
    ),
    "GatedDeltaNetOp": (
        "tests/ops/test_gated_deltanet_chunkwise_fwd.py",
        "test_gated_deltanet_fwd",
    ),
    "GLABwdOp": ("tests/ops/test_gla_chunkwise_bwd.py", "test_gla_bwd"),
    "GLADecodeOp": ("tests/ops/test_gla_recurrence.py", "test_gla_decode"),
    "GLAFwdOp": ("tests/ops/test_gla_chunkwise_fwd.py", "test_gla_fwd"),
    "GemmOp": ("tests/ops/test_gemm.py", "test_gemm"),
    "GqaSlidingWindowFwdOp": (
        "tests/ops/test_gqa_sliding_window_fwd.py",
        "test_gqa_sliding_window_fwd_op",
    ),
    "GqaSlidingWindowVarlenFwdOp": (
        "tests/ops/test_gqa_sliding_window_varlen_fwd.py",
        "test_gqa_sliding_window_varlen_fwd_op",
    ),
    "GroupNormOp": ("tests/ops/test_group_norm.py", "test_group_norm_op"),
    "GroupQueryAttentionBwdOp": ("tests/ops/test_gqa.py", "test_gqa_bwd"),
    "GroupQueryAttentionDecodePagedWithKVCacheOp": (
        "tests/ops/test_gqa_decode_paged.py",
        "test_gqa_decode_paged_op",
    ),
    "GroupQueryAttentionDecodeWithKVCacheOp": (
        "tests/ops/test_gqa_decode.py",
        "test_gqa_decode",
    ),
    "GroupQueryAttentionFwdOp": ("tests/ops/test_gqa.py", "test_gqa_fwd"),
    "GroupedGemmOp": ("tests/ops/test_grouped_gemm.py", "test_grouped_gemm"),
    "InfNormOp": ("tests/ops/test_vector_norm.py", "test_inf_norm_op"),
    "InstanceNormOp": ("tests/ops/test_instance_norm.py", "test_instance_norm_op"),
    "L1NormOp": ("tests/ops/test_vector_norm.py", "test_l1_norm_op"),
    "L2NormOp": ("tests/ops/test_vector_norm.py", "test_l2_norm_op"),
    "LayerNormOp": ("tests/ops/test_layer_norm.py", "test_layer_norm_op"),
    "LogSoftmaxOp": ("tests/ops/test_softmax.py", "test_log_softmax_op"),
    "LogSumExpOp": ("tests/ops/test_softmax.py", "test_logsumexp_op"),
    "ManifoldConstrainedHyperConnectionPostOp": (
        "tests/ops/test_mhc_post.py",
        "test_mhc_post_op",
    ),
    "ManifoldConstrainedHyperConnectionPreOp": (
        "tests/ops/test_mhc_pre.py",
        "test_mhc_pre_op",
    ),
    "MeanOp": ("tests/ops/test_reduce.py", "test_mean_op"),
    "MeanPoolingForwardOp": ("tests/ops/test_mean_pooling_ops.py", "test_mean_pooling_op"),
    "MoePermuteAlignOp": ("tests/ops/test_moe_permute_align.py", "test_permute_align_op"),
    "MultiHeadAttentionBwdOp": ("tests/ops/test_mha.py", "test_mha_bwd"),
    "MultiHeadAttentionDecodePagedWithKVCacheOp": (
        "tests/ops/test_mha_decode_paged.py",
        "test_mha_decode_paged_op",
    ),
    "MultiHeadAttentionDecodeWithKVCacheOp": (
        "tests/ops/test_mha_decode.py",
        "test_mha_decode",
    ),
    "MultiHeadAttentionFwdOp": ("tests/ops/test_mha.py", "test_mha_fwd"),
    "MultiHeadLatentAttentionDecodeWithKVCacheOp": (
        "tests/ops/test_deepseek_mla_decode.py",
        "test_mla_decode",
    ),
    "NSACmpFwdVarlenOp": (
        "tests/ops/test_deepseek_nsa_cmp_fwd.py",
        "test_nsa_cmp_fwd_varlen_op",
    ),
    "NSAFwdVarlenOp": ("tests/ops/test_deepseek_nsa_fwd.py", "test_nsa_varlen_op"),
    "NSATopkVarlenOp": ("tests/ops/test_deepseek_nsa_topk.py", "test_nsa_topk_varlen_op"),
    "ProdOp": ("tests/ops/test_reduce.py", "test_prod_op"),
    "RmsNormOp": ("tests/ops/test_rms_norm.py", "test_rms_norm_op"),
    "RopeLlama31Op": ("tests/ops/test_rope.py", "test_rope_llama31_1d"),
    "RopeLongRopeOp": ("tests/ops/test_rope.py", "test_rope_longrope_1d"),
    "RopeNeoxOp": ("tests/ops/test_rope.py", "test_rope_neox_1d"),
    "RopeNonNeoxOp": ("tests/ops/test_rope.py", "test_rope_non_neox_1d"),
    "RopeYarnOp": ("tests/ops/test_rope.py", "test_rope_yarn_1d"),
    "SoftmaxOp": ("tests/ops/test_softmax.py", "test_softmax_op"),
    "SsdChunkScanFwdOp": ("tests/ops/test_ssd_chunk_scan_fwd.py", "test_ssd_chunk_scan_fwd"),
    "SsdChunkStateFwdOp": (
        "tests/ops/test_ssd_chunk_state_fwd.py",
        "test_ssd_chunk_state_fwd",
    ),
    "SsdStatePassingFwdOp": (
        "tests/ops/test_ssd_state_passing_fwd.py",
        "test_ssd_state_passing_fwd",
    ),
    "StdOp": ("tests/ops/test_reduce.py", "test_std_op"),
    "SumOp": ("tests/ops/test_reduce.py", "test_sum_op"),
    "TopkSelectorOp": ("tests/ops/test_topk_selector.py", "test_topk_selector_op"),
    "UnaryOp": ("tests/ops/test_activation.py", "test_relu_op"),
    "VarMeanOp": ("tests/ops/test_reduce.py", "test_var_mean_op"),
    "VarOp": ("tests/ops/test_reduce.py", "test_var_op"),
}


def _load_public_op_names() -> list[str]:
    tree = ast.parse(OPS_INIT_PATH.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return sorted(
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and elt.value.endswith("Op")
                    and elt.value != "Op"
                )
    raise RuntimeError(f"Failed to load __all__ from {OPS_INIT_PATH}")


def _relative_test_path(pathlike: object) -> str:
    path = Path(str(pathlike)).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _set_explicit_tier(item: pytest.Item, tier: str) -> None:
    item.own_markers = [m for m in item.own_markers if m.name not in TIER_NAMES]
    item.add_marker(getattr(pytest.mark, tier))


def _clear_inherited_tiers_for_ops_items(items: list[pytest.Item]) -> None:
    visited_nodes: set[int] = set()
    for item in items:
        rel_path = _relative_test_path(item.path)
        if not rel_path.startswith(OPS_TEST_PREFIX):
            continue
        node = item.parent
        while node is not None:
            node_id = id(node)
            if node_id not in visited_nodes:
                node.own_markers = [m for m in node.own_markers if m.name not in TIER_NAMES]
                visited_nodes.add(node_id)
            node = node.parent


def _is_autotune_disabled(item: pytest.Item) -> bool:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return True
    for param_name in AUTOTUNE_PARAM_NAMES:
        if param_name in callspec.params:
            return callspec.params[param_name] is False
    return True


def _normalize_public_op_smoke(items: list[pytest.Item], tier_errors: list[str]) -> None:
    public_ops = _load_public_op_names()
    missing_targets = sorted(set(public_ops) - set(PUBLIC_OP_SMOKE_TARGETS))
    unexpected_targets = sorted(set(PUBLIC_OP_SMOKE_TARGETS) - set(public_ops))
    if missing_targets:
        tier_errors.append(
            "missing public Op smoke targets: " + ", ".join(missing_targets)
        )
        tier_errors.extend(
            [
                "add entries to PUBLIC_OP_SMOKE_TARGETS, for example:",
                *[
                    f'  "{op_name}": ("tests/ops/test_<op>.py", "test_<op>"),'
                    for op_name in missing_targets
                ],
            ]
        )
    if unexpected_targets:
        tier_errors.append(
            "unexpected public Op smoke targets: " + ", ".join(unexpected_targets)
        )
    if missing_targets or unexpected_targets:
        return

    _clear_inherited_tiers_for_ops_items(items)

    selected_smoke_items: dict[str, pytest.Item] = {}
    for op_name in public_ops:
        target_path, target_test_name = PUBLIC_OP_SMOKE_TARGETS[op_name]
        candidates = [
            item
            for item in items
            if _relative_test_path(item.path) == target_path
            and getattr(item, "originalname", item.name) == target_test_name
            and item.get_closest_marker("xfail") is None
        ]
        if not candidates:
            tier_errors.append(
                f"{op_name}: expected at least one non-xfail item at "
                f"{target_path}::{target_test_name}"
            )
            continue

        chosen_item = next((item for item in candidates if _is_autotune_disabled(item)), None)
        if chosen_item is None:
            tier_errors.append(
                f"{op_name}: all candidate smoke items in "
                f"{target_path}::{target_test_name} use autotune/tune=True"
            )
            continue
        selected_smoke_items[op_name] = chosen_item

    selected_nodeids = {item.nodeid for item in selected_smoke_items.values()}
    for item in items:
        rel_path = _relative_test_path(item.path)
        if not rel_path.startswith(OPS_TEST_PREFIX):
            continue
        if item.nodeid in selected_nodeids:
            _set_explicit_tier(item, "smoke")
        elif item.get_closest_marker("smoke") is not None or all(item.get_closest_marker(name) is None for name in TIER_NAMES):
            _set_explicit_tier(item, "full")

    smoke_counts: dict[str, int] = defaultdict(int)
    for op_name, item in selected_smoke_items.items():
        if item.get_closest_marker("smoke") is None:
            tier_errors.append(f"{op_name}: selected smoke item lost its smoke tier")
            continue
        if not _is_autotune_disabled(item):
            tier_errors.append(f"{item.nodeid}: smoke cases must use autotune/tune=False")
        smoke_counts[op_name] += 1

    for op_name in public_ops:
        if smoke_counts[op_name] != 1:
            tier_errors.append(
                f"{op_name}: expected exactly one smoke case, found {smoke_counts[op_name]}"
            )


@pytest.fixture(autouse=True)
def setup() -> None:
    torch.manual_seed(1235)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1235)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Validate explicit test tier assignments."""
    tier_errors: list[str] = []
    _normalize_public_op_smoke(items, tier_errors)

    ops_groups: dict[tuple[str, str], list[pytest.Item]] = defaultdict(list)
    for item in items:
        path = _relative_test_path(item.path)
        if not path.startswith(OPS_TEST_PREFIX):
            continue
        test_name = getattr(item, "originalname", item.name)
        ops_groups[(path, test_name)].append(item)

    for (_path, _test_name), group in ops_groups.items():
        first_tuned_item: pytest.Item | None = None
        full_tuned_items: list[pytest.Item] = []
        for item in group:
            callspec = getattr(item, "callspec", None)
            if callspec is None or "tune" not in callspec.params:
                continue

            tune = callspec.params["tune"]
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

    for item in items:
        path = _relative_test_path(item.path)
        if "tests/" not in path:
            continue
        tiers = [name for name in TIER_NAMES if item.get_closest_marker(name) is not None]
        if len(tiers) != 1:
            tier_errors.append(
                f"{item.nodeid}: expected exactly one tier marker, found {tiers or 'none'}"
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
