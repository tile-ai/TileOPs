"""Device lint distinguishes host metadata from CUDA placement and availability checks."""

import device_literal_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def test_a_cuda_availability_gate_is_flagged() -> None:
    source = "import torch\n\nif torch.cuda.is_available():\n    pass\n"
    assert lint.availability_findings(source) == [3]


def test_the_run_device_gate_passes() -> None:
    source = "from workloads.device import run_device_available\n\nok = run_device_available()\n"
    assert lint.availability_findings(source) == []


def test_an_architecture_mark_scopes_a_cuda_literal_as_cuda_only_does() -> None:
    source = (
        "import pytest\nimport torch\n\n\n@pytest.mark.sm90\ndef test_x():\n"
        '    torch.empty(1, device="cuda")\n'
    )
    assert lint.findings(source) == []


@pytest.mark.parametrize(
    "source, expected",
    [
        pytest.param(
            'import types\nx = types.SimpleNamespace(type="cuda", index=0)\n', [], id="module"
        ),
        pytest.param(
            'import types as t\nx = t.SimpleNamespace(type="cuda", index=0)\n',
            [],
            id="module-alias",
        ),
        pytest.param(
            'from types import SimpleNamespace\nx = SimpleNamespace(type="cuda", index=0)\n',
            [],
            id="constructor",
        ),
        pytest.param(
            'from types import SimpleNamespace as NS\nx = NS(type="cuda", index=0)\n',
            [],
            id="constructor-alias",
        ),
        pytest.param(
            'from types import SimpleNamespace\nx = SimpleNamespace(type="cuda", device="cuda")\n',
            [2],
            id="other-keyword",
        ),
        pytest.param(
            'from types import SimpleNamespace\ndef test_x():\n    x = SimpleNamespace(type="cuda", index=0)\n    y = torch.empty(1, device="cuda")\n',
            [4],
            id="tensor-in-same-function",
        ),
        pytest.param(
            'from types import SimpleNamespace\nx = SimpleNamespace(type=torch.empty(1, device="cuda"))\n',
            [2],
            id="nested-allocation",
        ),
        pytest.param('x = Unknown(type="cuda")\n', [1], id="unknown-call"),
        pytest.param('x = SimpleNamespace(type="cuda")\n', [1], id="unimported-constructor"),
        pytest.param('import torch\nx = torch.device("cuda")\n', [2], id="device-object"),
        pytest.param(
            'import types\nx = types.SimpleNamespace(type="cuda:0")\n', [2], id="not-a-device-type"
        ),
    ],
)
def test_only_imported_namespace_type_metadata_is_exempt(source: str, expected: list[int]) -> None:
    assert lint.findings(source) == expected
