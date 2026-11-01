"""An op builds for the target of its call's device (docs/design/ops-design.md § Kernel
selection).

A device is described as an architecture of another major version through the device-facts
reader (``tests.build_target_facts``). A call there compiles its own program, which the device
refuses to load: build and dispatch are isolated by target. Whether a kernel runs correctly on
that architecture is tested on such a device.
"""

import pytest
import torch

from tests.build_target_facts import arch, compiles, described, other_arch, sm
from tests.workload_test_base import TestBase
from tileops.norm import RMSNormFwdOp
from workloads.norm import RMSNormWorkload

pytestmark = [pytest.mark.cuda_only, pytest.mark.in_tree_kernels, pytest.mark.smoke]


def test_an_op_on_another_target_builds_its_own_program():
    """A call on a device of another target compiles for that target; a return to the first
    target compiles nothing and computes the reference."""
    workload = RMSNormWorkload(64, 1024, torch.float16)
    inputs = workload.gen_inputs()
    index = inputs[0].device.index
    with compiles() as built:
        TestBase.check(workload, RMSNormFwdOp((1024,), eps=workload.eps), *inputs)
        real = list(built)
        with (
            described({index: {"arch": other_arch(index)}}),
            pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"),
        ):
            RMSNormFwdOp((1024,), eps=workload.eps)(*inputs)
        other = built[len(real) :]
        TestBase.check(workload, RMSNormFwdOp((1024,), eps=workload.eps), *inputs)
    # The real target's programs may already be built by an earlier test in this process.
    assert set(real) <= {sm(arch(index))}
    assert other and set(other) == {sm(other_arch(index))}
    assert len(built) == len(real) + len(other)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second CUDA device")
def test_one_op_on_two_targets_builds_for_each():
    """One op called on devices of two targets compiles for each, a first launch on the second
    while the first is current included, and each repeat reuses its own target's program."""
    # Another shape than the other test, so no build of the described target precedes this one.
    workload = RMSNormWorkload(32, 2048, torch.float16)
    first = [t.to("cuda:0") for t in workload.gen_inputs()]
    second = [t.to("cuda:1") for t in first]
    op = RMSNormFwdOp((2048,), eps=workload.eps)
    with (
        described({1: {"arch": other_arch(1)}}),
        torch.accelerator.device_index(0),
        compiles() as built,
    ):
        TestBase.check(workload, op, *first)
        real = list(built)
        with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
            op(*second)
        other = built[len(real) :]
        TestBase.check(workload, op, *first)
        with pytest.raises(RuntimeError, match="NO_BINARY_FOR_GPU"):
            op(*second)
    assert set(real) <= {sm(arch(0))}
    assert other and set(other) == {sm(other_arch(1))}
    assert len(built) == len(real) + len(other)
