"""The PrivateUse1 test backend keeps a tensor's bytes on its own device and copies its views."""

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


def test_a_copy_into_an_empty_tensor_checks_the_shapes():
    with pytest.raises(RuntimeError, match="must match the size"):
        torch.empty(0, device=run_device()).copy_(torch.ones(2))


def test_a_device_generator_draws_what_the_host_generator_draws(two_devices):
    _, second = two_devices
    drawn = torch.randn(4, device=second, generator=torch.Generator(second).manual_seed(5))
    expected = torch.randn(4, generator=torch.Generator().manual_seed(5))
    assert drawn.device == second
    assert torch.equal(drawn.cpu(), expected)


def test_a_convolution_and_its_gradients_match_the_host():
    host = [torch.randn(1, 2, 8), torch.randn(3, 2, 3), torch.randn(3)]
    on_device = [t.to(run_device()).requires_grad_() for t in host]
    for t in host:
        t.requires_grad_()
    expected = torch.nn.functional.conv1d(*host)
    got = torch.nn.functional.conv1d(*on_device)
    expected.sum().backward()
    got.sum().backward()
    assert torch.allclose(got.detach().cpu(), expected.detach())
    for t, u in zip(on_device, host, strict=True):
        assert torch.allclose(t.grad.cpu(), u.grad)


def test_a_device_reports_free_and_total_memory():
    free, total = torch.accelerator.get_memory_info(0)
    assert 0 < free <= total
