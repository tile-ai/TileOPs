"""The public modules resolve from wherever `tileops` is installed.

`scripts/ci/verify_installed_wheel.sh` runs this file against the installed wheel, so a
public module or name the wheel leaves out fails there, not on a user's machine.
"""

import importlib

import pytest

import tileops

_PUBLIC_MODULES = (
    "tileops",
    *(f"tileops.{family}" for family in tileops._FAMILIES),
    "tileops.authoring",
    "tileops.backend",
    "tileops.manifest",
    "tileops.manifest.primitives",
    "tileops.trace",
)


@pytest.mark.smoke
@pytest.mark.packaging
@pytest.mark.parametrize("name", _PUBLIC_MODULES)
def test_every_listed_name_resolves(name: str):
    mod = importlib.import_module(name)
    assert [attr for attr in getattr(mod, "__all__", ()) if not hasattr(mod, attr)] == []


@pytest.mark.smoke
@pytest.mark.packaging
def test_an_op_outside_tileops_defines_and_constructs():
    from tileops.authoring import Op

    class NegateFwdOp(Op):
        spec = {
            "family": "elementwise",
            "signature": {
                "forall": {"S": "Shape", "T": "DType[float32]"},
                "inputs": {"input": {"dtype": "T", "shape": "[*S]"}},
                "outputs": {"output": {"dtype": "T", "shape": "[*S]"}},
            },
        }

        def __init__(self, *, target=None):
            super().__init__(target=target)

        def forward(self, input):
            return -input

    assert Op is tileops.Op
    assert isinstance(NegateFwdOp(), Op)
