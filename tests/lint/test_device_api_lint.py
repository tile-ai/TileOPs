"""The device API lint flags torch.cuda on the path every device runs through."""

import device_api_lint as lint
import pytest

pytestmark = pytest.mark.smoke


def test_a_torch_cuda_call_is_flagged() -> None:
    source = "import torch\n\ntorch.cuda.synchronize()\n"
    assert lint.findings(source) == [3]


def test_cuda_only_code_with_a_reason_and_the_accelerator_api_pass() -> None:
    source = (
        "import torch\n\n"
        "torch.cuda.synchronize()  # lint: device-api allow: CUPTI runs on CUDA only\n"
        "torch.accelerator.synchronize()\n"
    )
    assert lint.findings(source) == []


def test_a_marker_without_a_reason_exempts_nothing() -> None:
    assert lint.findings("torch.cuda.synchronize()  # lint: device-api allow:\n") == [1]


def test_only_the_shared_path_is_scanned(tmp_path) -> None:
    kernel = tmp_path / "kernel.py"
    kernel.write_text("torch.cuda.synchronize()\n")
    assert lint.file_findings(kernel) == []


@pytest.mark.parametrize(
    "source",
    ["from torch.cuda import synchronize\n", "from torch import cuda\n", "import torch.cuda\n"],
    ids=["from-module", "from-torch", "import-module"],
)
def test_an_import_of_torch_cuda_is_flagged(source: str) -> None:
    assert lint.findings(source) == [1]
