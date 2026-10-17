"""The baseline import lint flags a bench file importing a baseline library itself."""

import baseline_import_lint as lint
import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "source",
    [
        "import flashinfer\n",
        "from fla.ops.gla import chunk_gla\n",
        "def f():\n    try:\n        from vllm import _custom_ops\n    except ImportError:\n        pass\n",
    ],
    ids=["module", "from-submodule", "guarded-in-function"],
)
def test_an_import_of_a_baseline_library_is_flagged(source: str) -> None:
    assert len(lint.findings(source)) == 1


def test_the_harness_tools_and_the_resolvers_pass() -> None:
    source = (
        "import torch\nimport triton\n"
        "from benchmarks.baselines import optional_baseline\n"
        'chunk_gla = optional_baseline("fla.ops.gla", "fla", "chunk_gla")\n'
    )
    assert lint.findings(source) == []
