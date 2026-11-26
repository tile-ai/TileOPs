"""``DropoutKernel`` refuses a thread run its Philox draws do not fit."""

import pytest
import torch

from tileops._kernels.elementwise.dropout import DropoutKernel

pytestmark = [pytest.mark.cuda_only("cuda-kernel")]


@pytest.mark.parametrize(
    "num_per_thread",
    [
        pytest.param(2, marks=pytest.mark.smoke),
        pytest.param(64, marks=pytest.mark.smoke),
    ],
)
def test_dropout_kernel_rejects_a_thread_run_off_its_draws(num_per_thread: int) -> None:
    """A thread run that is not a multiple of 4 leaves Philox words unwritten or writes past
    them, and one above ``MAX_DRAWS_PER_STREAM`` draws past the generator's advance."""
    with pytest.raises(ValueError, match="num_per_thread"):
        DropoutKernel(1024, torch.float16, config={"num_per_thread": num_per_thread})
