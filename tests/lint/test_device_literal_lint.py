"""The device lint flags a CUDA availability gate."""

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
