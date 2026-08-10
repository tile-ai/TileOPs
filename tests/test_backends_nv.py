"""The nv backend as seen from outside: it registers like any other, with no privilege."""

import subprocess
import sys

import pytest
import torch

from tileops.backends import nv
from tileops.backends.nv._bindings import BINDINGS


@pytest.mark.smoke
def test_installing_this_distribution_is_enough_to_find_nv():
    """Found by enumerating the entry point group, without anyone importing nv.

    Checked in a fresh interpreter: importing ``tileops.backends.nv`` here first would prove
    only that the module registers when imported, which is not what installing a backend has
    to do.
    """
    probe = "import tileops.backend as b; print(sorted(b.registrations()))"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "('RMSNormFwdOp', 'nv')" in result.stdout


@pytest.mark.smoke
def test_nv_claims_nvidia_gpus_and_nothing_else():
    assert nv.detect(torch.device("cpu")) is False
    assert nv.detect(torch.device("cuda", 0)) is (torch.version.hip is None)


@pytest.mark.smoke
def test_a_rocm_build_is_not_nv_even_though_torch_calls_it_cuda(monkeypatch):
    """These kernels do not run there, and `device.type` alone cannot tell the two apart."""
    monkeypatch.setattr(torch.version, "hip", "6.0.0")
    assert nv.detect(torch.device("cuda", 0)) is False


@pytest.mark.smoke
def test_importing_nv_pulls_in_no_kernel_and_no_tilelang():
    """Its bindings are strings, so a kernel module is imported when an op is built."""
    probe = (
        "import sys, tileops.backends.nv as nv; "
        "print(any(m.startswith(('tilelang', 'tileops.kernels')) for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


@pytest.mark.smoke
def test_bindings_are_this_backend_s_own_data():
    """Not read from the neutral manifest, which a third-party backend cannot write to."""
    from tileops.manifest import load_manifest

    assert BINDINGS, "the table is the backend's data, and it is not empty"
    for entry in load_manifest().values():
        assert "kernel_map" not in (entry.get("source") or {})


@pytest.mark.parametrize("op", sorted(BINDINGS))
@pytest.mark.full
def test_every_binding_names_a_class_that_exists(op):
    """A rename in the kernel layer must not leave this table quietly pointing at nothing."""
    from tileops.backends.nv._load import kernel_class

    for role in BINDINGS[op]:
        assert isinstance(kernel_class(op, role), type)
