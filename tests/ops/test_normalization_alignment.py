"""Manifest-alignment conformance tests for the ``normalization`` family.

Covers the 10 normalization ops covered by the family alignment
(RMSNormFwdOp, LayerNormFwdOp, BatchNormFwdOp, BatchNormBwdOp,
GroupNormFwdOp, InstanceNormFwdOp, AdaLayerNormFwdOp,
AdaLayerNormZeroFwdOp, FusedAddLayerNormFwdOp, FusedAddRMSNormFwdOp).

Each test asserts the live Op class signature satisfies the manifest L1
contract: every manifest input appears in ``forward()`` in declaration
order, every manifest param is reachable through ``__init__`` or
``forward``. Test data is read from
``tileops.manifest.load_manifest()`` so manifest changes flow into the
assertions automatically.
"""

from __future__ import annotations

import pytest

from tileops.manifest import load_manifest

# Op-class import paths for each manifest op in this family.
_OP_IMPORTS: dict[str, str] = {
    "RMSNormFwdOp": "tileops.ops.norm.rms_norm",
    "LayerNormFwdOp": "tileops.ops.norm.layer_norm",
    "AdaLayerNormFwdOp": "tileops.ops.norm.ada_layer_norm",
    "AdaLayerNormZeroFwdOp": "tileops.ops.norm.ada_layer_norm_zero",
    "FusedAddLayerNormFwdOp": "tileops.ops.norm.fused_add_layer_norm",
    "FusedAddRMSNormFwdOp": "tileops.ops.norm.fused_add_rms_norm",
    "BatchNormFwdOp": "tileops.ops.norm.batch_norm",
    "BatchNormBwdOp": "tileops.ops.norm.batch_norm",
    "GroupNormFwdOp": "tileops.ops.norm.group_norm",
    "InstanceNormFwdOp": "tileops.ops.norm.instance_norm",
}

_NORMALIZATION_OPS = tuple(_OP_IMPORTS.keys())

_MANIFEST = load_manifest()


def _signature_case(op_name: str):
    entry = _MANIFEST[op_name]["signature"]
    return (
        op_name,
        entry.get("inputs", {}),
        entry.get("params", {}),
    )


_SIGNATURE_CASES = [_signature_case(n) for n in _NORMALIZATION_OPS]


def _import_op(op_name: str):
    import importlib

    mod = importlib.import_module(_OP_IMPORTS[op_name])
    return getattr(mod, op_name)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_name, manifest_inputs, manifest_params",
    _SIGNATURE_CASES,
    ids=[c[0] for c in _SIGNATURE_CASES],
)
def test_normalization_signature_matches_manifest(
    op_name: str,
    manifest_inputs: dict,
    manifest_params: dict,
) -> None:
    """Op class signatures must satisfy the manifest L1 contract."""
    from scripts.validate_manifest import (
        _get_forward_params,
        _get_init_params,
        check_l1_signature,
    )

    cls = _import_op(op_name)
    forward_params = _get_forward_params(cls)
    assert forward_params is not None, (
        f"Cannot extract forward() params for {op_name}"
    )
    init_params = _get_init_params(cls)
    errors = check_l1_signature(
        op_name,
        manifest_inputs,
        manifest_params,
        forward_params,
        init_params=init_params,
    )
    assert errors == [], f"{op_name}: {errors}"


# ---------------------------------------------------------------------------
# Output-arity contract: BatchNormFwdOp must return a single tensor (manifest
# outputs: {output}); the older API returned (y, mean, rstd).
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_batch_norm_fwd_returns_single_output() -> None:
    """BatchNormFwdOp manifest declares one output; forward returns one tensor."""
    import inspect

    from tileops.ops.norm.batch_norm import BatchNormFwdOp

    sig = inspect.signature(BatchNormFwdOp.forward)
    return_ann = sig.return_annotation
    # Manifest outputs == {output}, so forward must not return a tuple.
    # The runtime contract is exercised in tests/ops/test_batch_norm.py.
    assert return_ann is not inspect.Signature.empty
    ann_str = str(return_ann)
    assert "Tuple" not in ann_str and "tuple" not in ann_str, (
        f"BatchNormFwdOp.forward annotation {ann_str} must be a single tensor"
    )


# ---------------------------------------------------------------------------
# RMSNorm/LayerNorm: ``normalized_shape`` ctor path is the manifest contract.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_rms_norm_accepts_normalized_shape() -> None:
    import torch

    from tileops.ops.norm.rms_norm import RMSNormFwdOp

    op = RMSNormFwdOp(normalized_shape=(4096,), eps=None, dtype=torch.float16)
    assert op.N == 4096
    assert op.normalized_shape == (4096,)


@pytest.mark.smoke
def test_layer_norm_accepts_normalized_shape() -> None:
    import torch

    from tileops.ops.norm.layer_norm import LayerNormFwdOp

    op = LayerNormFwdOp(normalized_shape=[4096], dtype=torch.float16)
    assert op.N == 4096
    assert op.normalized_shape == (4096,)


# ---------------------------------------------------------------------------
# GroupNormFwdOp: ``num_groups`` is the manifest-aligned ctor name.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_group_norm_uses_num_groups_kwarg() -> None:
    import torch

    from tileops.ops.norm.group_norm import GroupNormFwdOp

    op = GroupNormFwdOp(
        N=2, C=32, spatial=(8, 8), num_groups=8, dtype=torch.float16,
    )
    assert op.num_groups == 8


# ---------------------------------------------------------------------------
# InstanceNormFwdOp: running-stats variant (use_input_stats=False) is OOS.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_instance_norm_rejects_use_input_stats_false() -> None:
    import torch

    from tileops.ops.norm.instance_norm import InstanceNormFwdOp

    with pytest.raises(NotImplementedError):
        InstanceNormFwdOp(
            N=2, C=16, spatial=(8, 8), dtype=torch.float16,
            use_input_stats=False,
        )
