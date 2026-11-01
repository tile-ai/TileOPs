"""The PrivateUse1 test backend keeps a tensor's bytes on its own device and copies its views.

Runs where ``-p tests.privateuse1`` made that backend the run device, and skips elsewhere.
"""

import pytest
import torch

from workloads.device import run_device

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        torch.device(run_device()).type != "tileops_test",
        reason="tests the tileops_test backend, which this run does not use",
    ),
]


def test_a_tensor_grows_on_its_own_device_whichever_is_current(two_devices):
    first, second = two_devices
    x = torch.ones(3, device=second)

    with torch.accelerator.device_index(first.index):
        x.resize_(8)

    assert x.untyped_storage().device == second
    assert (x[:3] + 1).device == second


@pytest.mark.parametrize(
    ("view", "expected"),
    [
        (lambda t: t.conj(), torch.tensor([1 - 2j])),
        (lambda t: torch._neg_view(t), torch.tensor([-1 - 2j])),
    ],
    ids=["conjugate", "negative"],
)
def test_a_lazy_view_reaches_the_host_resolved(view, expected):
    source = torch.tensor([1 + 2j], device=run_device())
    assert torch.equal(view(source).cpu(), expected)
