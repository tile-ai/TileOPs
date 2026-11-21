"""The device lints: no CUDA literal or availability gate, and no torch.cuda on the shared path."""

import re

import device_api_lint
import device_literal_lint
import pytest
import yaml
from common import REPO_ROOT

pytestmark = pytest.mark.smoke


def test_a_cuda_availability_gate_is_flagged() -> None:
    source = "import torch\n\nif torch.cuda.is_available():\n    pass\n"
    assert device_literal_lint.availability_findings(source) == [3]


def test_the_run_device_gate_passes() -> None:
    source = "from workloads.device import run_device_available\n\nok = run_device_available()\n"
    assert device_literal_lint.availability_findings(source) == []


def test_an_architecture_mark_scopes_a_cuda_literal_as_cuda_only_does() -> None:
    source = (
        "import pytest\nimport torch\n\n\n@pytest.mark.sm90\ndef test_x():\n"
        '    torch.empty(1, device="cuda")\n'
    )
    assert device_literal_lint.findings(source) == []


def test_an_allowance_with_a_reason_exempts_a_literal_that_places_nothing() -> None:
    source = (
        'device = SimpleNamespace(type="cuda", index=0)  # lint: device-literal allow: no tensor\n'
    )
    assert device_literal_lint.findings(source) == []


def test_a_torch_cuda_call_is_flagged() -> None:
    source = "import torch\n\ntorch.cuda.synchronize()\n"
    assert device_api_lint.findings(source) == [3]


def test_cuda_only_code_with_a_reason_and_the_accelerator_api_pass() -> None:
    source = (
        "import torch\n\n"
        "torch.cuda.synchronize()  # lint: device-api allow: CUPTI runs on CUDA only\n"
        "torch.accelerator.synchronize()\n"
    )
    assert device_api_lint.findings(source) == []


def test_a_marker_without_a_reason_exempts_nothing() -> None:
    assert device_api_lint.findings("torch.cuda.synchronize()  # lint: device-api allow:\n") == [1]


def test_only_the_shared_path_is_scanned(tmp_path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("torch.cuda.synchronize()\n")
    assert device_api_lint.file_findings(kernel) == []


@pytest.mark.parametrize(
    "source",
    ["from torch.cuda import synchronize\n", "from torch import cuda\n", "import torch.cuda\n"],
    ids=["from-module", "from-torch", "import-module"],
)
def test_an_import_of_torch_cuda_is_flagged(source: str) -> None:
    assert device_api_lint.findings(source) == [1]


def test_the_hook_runs_on_every_scanned_file() -> None:
    """pre-commit hands the hook the files its pattern matches, so a scanned file the pattern
    misses is never checked."""
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    (hook,) = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "device-api-lint"
    ]
    pattern = re.compile(hook["files"])
    scanned = [p for path in device_api_lint.SCANNED for p in _files(REPO_ROOT / path)]
    assert scanned and [p for p in scanned if not pattern.search(p)] == []


def _files(path):
    """The repository-relative Python files at *path*, a file or a tree."""
    found = sorted(path.rglob("*.py")) if path.is_dir() else [path]
    return [p.relative_to(REPO_ROOT).as_posix() for p in found]
