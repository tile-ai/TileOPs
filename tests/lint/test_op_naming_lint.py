"""Unit tests for ``scripts/lint/op_naming_lint.py``."""

import op_naming_lint as lint
import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "ops, expected",
    [
        pytest.param({"MoeFwdOp": {}}, "M1 MoeFwdOp: spell this abbreviation MoE", id="m1"),
        pytest.param({"TopkSelectorFwdOp": {}}, "M2 TopkSelectorFwdOp: 'Selector'", id="m2"),
        pytest.param(
            {
                "SoftmaxFwdOp": {"ref_api": "torch.softmax"},
                "SoftmaxOnlineFwdOp": {"ref_api": "torch.softmax"},
            },
            "M3 SoftmaxFwdOp: torch.softmax has no default side",
            id="m3",
        ),
    ],
)
def test_an_entry_name_a_rule_reaches_is_reported(ops: dict, expected: str) -> None:
    assert any(expected in f for f in lint.findings(ops)), lint.findings(ops)


def test_a_finding_names_the_line_its_entry_starts_on() -> None:
    (found,) = lint.findings({"MlpFwdOp": {}}, {"MlpFwdOp": "spec/mlp.yaml:7"})

    assert found.startswith("spec/mlp.yaml:7: M1 MlpFwdOp:")


def test_an_allowed_or_defaulted_name_is_left_alone() -> None:
    ops = {
        **{name: {} for name in lint.AGENTIVE_EXCEPTIONS},
        "MaxPool2dFwdOp": {"ref_api": "torch.nn.functional.max_pool2d"},
        "MaxPool2dIndicesFwdOp": {"ref_api": "torch.nn.functional.max_pool2d"},
    }

    assert lint.findings(ops) == []


def test_every_manifest_entry_has_a_location() -> None:
    assert set(lint.load_manifest()) <= set(lint.entry_locations())
