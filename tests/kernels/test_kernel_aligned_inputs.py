"""``Kernel.aligned_inputs``: the copy ``Kernel.__call__`` makes of a vector-loaded input."""

import pytest
import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES
from tileops.kernels.kernel_base import Kernel

pytestmark = pytest.mark.smoke


class _Recorder(Kernel):
    aligned_inputs = ("x", "y")

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> tuple:
        return x, y, z


def _off_boundary() -> torch.Tensor:
    """A contiguous view starting one element past a 16-byte boundary."""
    return torch.arange(9, dtype=torch.float16)[1:]


def test_declared_inputs_off_a_vector_boundary_arrive_copied() -> None:
    x, y, z = _off_boundary(), _off_boundary(), _off_boundary()
    got_x, got_y, got_z = _Recorder()(x, z=z, y=y)
    # Declared, positional or keyword: a copy with the same values on the boundary.
    for sent, got in ((x, got_x), (y, got_y)):
        assert got is not sent and torch.equal(got, sent)
        assert got.data_ptr() % VECTOR_ACCESS_BYTES == 0
    # Undeclared: handed over as is.
    assert got_z is z

    narrow = _Recorder()
    narrow.aligned_inputs = ()
    assert narrow(x, y, z)[0] is x

    with pytest.raises(TypeError, match="names no forward parameter"):

        class _Typo(Kernel):
            aligned_inputs = ("w",)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x
