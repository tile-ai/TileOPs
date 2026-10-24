"""What a repeated eager call observes of its launch (docs/design/ops-design.md § Call protocol).

A repeated specialization reuses its plan and its kernel's programs; the addresses, the
stream and the outputs belong to each call.
"""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.backend import BUILTIN
from tileops.ops import GemmFwdOp
from tileops.ops.elementwise import SiluFwdOp
from workloads.device import run_device
from workloads.elementwise import UnaryActivationCase
from workloads.gemm import GemmWorkload
from workloads.numerics import compare_outputs

pytestmark = [pytest.mark.smoke, pytest.mark.cuda_only, pytest.mark.in_tree_kernels]

_N = 4096


class _SiluTest(UnaryActivationCase, TestBase):
    pass


class _GemmTest(GemmWorkload, TestBase):
    pass


_CASE = _SiluTest(_N, torch.bfloat16, "SiluFwdOp")


def _verify(y: torch.Tensor, x: torch.Tensor) -> None:
    """Hold *y* to the workload's reference for *x* under its verification."""
    compare_outputs(y, _CASE.ref_program(x), _CASE.verification(x))


def test_a_warm_call_on_a_misaligned_slice_of_the_same_layout_is_correct():
    """An equal shape, stride, dtype and device does not fix the address: a slice starting
    off a vector boundary is prepared on its own launch."""
    op = SiluFwdOp(target=BUILTIN)
    base = torch.randn(_N + 1, dtype=torch.bfloat16, device=run_device())
    aligned = base[:_N].clone()
    op(aligned)
    shifted = base[1:]
    assert shifted.shape == aligned.shape and shifted.stride() == aligned.stride()
    _verify(op(shifted), shifted)


def test_a_warm_call_runs_on_the_current_stream():
    """The stream is the caller's on each launch: work queued before the call on a side
    stream is ordered before the kernel."""
    op = SiluFwdOp(target=BUILTIN)
    x = torch.zeros(_N, dtype=torch.bfloat16, device=run_device())
    op(x)
    side = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)
        x.fill_(1.0)
        y = op(x)
    side.synchronize()
    _verify(y, x)


def test_outputs_kept_alive_are_not_reused_by_later_calls():
    """Each call returns a fresh output: two earlier results stay as they were."""
    op = SiluFwdOp(target=BUILTIN)
    inputs = [torch.full((_N,), v, dtype=torch.bfloat16, device=run_device()) for v in (1, 2, 3)]
    outputs = [op(x) for x in inputs]
    for x, y in zip(inputs, outputs, strict=True):
        _verify(y, x)


def test_a_captured_warm_call_replays_on_updated_inputs():
    """A warm call captures into a CUDA graph and replays on the inputs' new contents."""
    op = SiluFwdOp(target=BUILTIN)
    x = torch.zeros(_N, dtype=torch.bfloat16, device=run_device())
    op(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = op(x)
    x.fill_(2.0)
    graph.replay()
    torch.cuda.synchronize()
    _verify(y, x)


def test_a_published_decomposition_serves_the_next_call_and_its_capture():
    """A configuration naming another decomposition is launched from its own programs on
    the next call, and a graph captured afterwards replays them."""
    test = _GemmTest(256, 256, 512, torch.bfloat16, False, True)
    a, b = test.gen_inputs()
    op = GemmFwdOp(trans_b=True, target=BUILTIN)
    op(a, b)
    (kernel,) = op.iter_kernels()
    warmed = kernel._generation  # held, so no new program takes the id of one of its own
    warm = set(map(id, warmed.programs.values()))
    simple = {"simple": True, "block_m": 64, "block_n": 64, "block_k": 64, "num_stages": 2}
    kernel.config = {**kernel.config, **simple}
    published = kernel._generation
    test.check(op, a, b)
    # The call compiled its launch into the published generation, from nothing held before.
    launched = set(map(id, published.programs.values()))
    assert launched and not launched & warm
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = op(a, b)
    a.copy_(torch.randn_like(a))
    graph.replay()
    torch.cuda.synchronize()
    compare_outputs(y, test.ref_program(a, b), test.verification(a, b))
