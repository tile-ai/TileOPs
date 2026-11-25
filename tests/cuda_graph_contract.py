"""Ops with H200 CUDA Graph evidence, and the capture and replay their evidence shares; marked
tests define the qualified calls."""

from collections.abc import Callable, Collection, Iterator, Sequence
from typing import Any

import torch

from tests.workload_test_base import TestBase, _by_signature

QUALIFIED_OPS = frozenset(
    {
        "ArgmaxFwdOp",
        "BMMFwdOp",
        "CausalConv1dFwdOp",
        "CausalConv1dUpdateFwdOp",
        "Conv1dFwdOp",
        "EmbeddingFwdOp",
        "FP4QuantPerGroupFwdOp",
        "FP8QuantPerTokenGroupFwdOp",
        "FusedMoEExpertsFwdOp",
        "GEMMFP8FwdOp",
        "GEMMFwdOp",
        "GEMMW4A8FwdOp",
        "GQAPagedFwdOp",
        "GQAVarlenFwdOp",
        "HadamardTransformFwdOp",
        "HyperConnectionPreFwdOp",
        "KDAFwdOp",
        "MergeAttentionStatesFwdOp",
        "MulFwdOp",
        "NGramHashFwdOp",
        "PagedKVCacheGatherFwdOp",
        "ReLUFwdOp",
        "RMSNormFwdOp",
        "RoPEFwdOp",
        "SigmoidFwdOp",
        "SiLUAndMulFwdOp",
    }
)


class CapturedGraph:
    """*calls* captured back to back into one CUDA graph, each an op, or a function, and its
    inputs, an op's in signature order; ``outputs`` holds what each call returned.

    Each call first runs on a side stream on copies of its inputs at other addresses, so no
    call may reach the inputs before the capture. The copies hold the captured values as long as the
    graph lives: an op that keeps anything of its first call's tensors reads those values at
    replay, not the ones the test wrote.
    """

    def __init__(self, *calls: tuple[Callable[..., Any], Sequence[Any]]) -> None:
        self._calls = [(fn, tuple(inputs)) for fn, inputs in calls]
        storages: dict[int, tuple[torch.UntypedStorage, torch.UntypedStorage]] = {}
        copies = [[_copy(x, storages) for x in inputs] for _, inputs in self._calls]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for (fn, _), inputs in zip(self._calls, copies, strict=True):
                _by_signature(fn, fn)(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        for storage, copy in storages.values():
            copy.copy_(storage)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.outputs = [_by_signature(fn, fn)(*inputs) for fn, inputs in self._calls]
        self._inputs: dict[int, tuple[str, torch.Tensor, torch.Tensor]] = {}
        for index, ((_, inputs), own) in enumerate(zip(self._calls, copies, strict=True)):
            for position, (x, copy) in enumerate(zip(inputs, own, strict=True)):
                if isinstance(x, torch.Tensor):
                    self._inputs.setdefault(id(x), (f"call {index} input {position}", x, copy))
        read = {x.untyped_storage().data_ptr() for _, x, _ in self._inputs.values()}
        self._poisoned = [
            t for t in _tensors(self.outputs) if t.untyped_storage().data_ptr() not in read
        ]

    def renew(self, *fresh: Sequence[Any]) -> None:
        """Copy *fresh*, one input sequence a call, into each call's tensors at their captured
        addresses."""
        for (_, inputs), values in zip(self._calls, fresh, strict=True):
            for x, value in zip(inputs, values, strict=True):
                if isinstance(x, torch.Tensor):
                    x.copy_(value)

    def replay(self, held: Collection[torch.Tensor] = ()) -> None:
        """Fill each output that is no input with NaN, or where it holds integers with its
        dtype's minimum, since several ops fill or pad with -1; then replay.

        Raises:
            AssertionError: An input other than the *held* ones holds its captured value, which
                hides a read of anything but its captured address. A test holds an input only
                for a reason it states.
        """
        unchanged = [
            name
            for name, x, copy in self._inputs.values()
            if all(x is not h for h in held) and torch.equal(_bits(x), _bits(copy))
        ]
        assert not unchanged, f"rewrite or hold each input at its captured value: {unchanged}"
        for tensor in self._poisoned:
            tensor.fill_(
                float("nan") if tensor.is_floating_point() else torch.iinfo(tensor.dtype).min
            )
        self._graph.replay()


def check_replay(
    test: Any,
    op: Callable[..., Any],
    fresh: Sequence[Any] | None = None,
    held: Collection[int] = (),
    result: Callable[..., Any] | None = None,
) -> None:
    """Capture *op* on a draw of *test*'s inputs, renew them from *fresh*, another draw unless
    given, replay with the *held* positions exempt, and check the replay against *test*'s
    reference on *fresh*. *result*, given the replay and the captured inputs, forms what an op
    that writes its inputs is compared on."""
    inputs = test.gen_inputs()
    graph = CapturedGraph((op, inputs))
    fresh = test.gen_inputs() if fresh is None else fresh
    graph.renew(fresh)
    graph.replay(held=[inputs[position] for position in held])
    output = graph.outputs[0] if result is None else result(lambda *_: graph.outputs[0], *inputs)
    TestBase.check(test, op, *fresh, runs=lambda *args, **kwargs: output)


def _copy(x: Any, storages: dict) -> Any:
    """*x* if it is no tensor, else *x*'s view of a copy of its storage, one copy a storage."""
    if not isinstance(x, torch.Tensor):
        return x
    storage = x.untyped_storage()
    if storage.data_ptr() not in storages:
        storages[storage.data_ptr()] = storage, storage.clone()
    copy = storages[storage.data_ptr()][1]
    return x.new_empty(0).set_(copy, x.storage_offset(), x.shape, x.stride())


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    """The tensors in *value*, a tensor or sequences of them."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """*x*'s bytes, which compare for every dtype and NaN alike."""
    return x.reshape(-1).contiguous().view(torch.uint8)
