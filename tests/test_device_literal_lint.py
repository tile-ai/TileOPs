"""The device lint flags a CUDA availability gate."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lint" / "device_literal_lint.py"
_SPEC = importlib.util.spec_from_file_location("device_literal_lint", _PATH)
lint = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lint)


def test_a_cuda_availability_gate_is_flagged() -> None:
    source = "import torch\n\nif torch.cuda.is_available():\n    pass\n"
    assert lint.availability_findings(source) == [3]


def test_the_run_device_gate_passes() -> None:
    source = "from workloads.device import run_device_available\n\nok = run_device_available()\n"
    assert lint.availability_findings(source) == []
