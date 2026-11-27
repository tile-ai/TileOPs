"""``CausalLM`` on a model whose forward body is plain tensor arithmetic."""

from types import SimpleNamespace

import pytest
import torch

from models.causal_lm import CausalLM

pytestmark = pytest.mark.nightly

_VOCAB = 8


class _Model(CausalLM):
    """Logits that pick the last token's id plus one, mod 8, each offset by the call's start
    position."""

    def __init__(self, max_seq_len: int = 16) -> None:
        self.config = SimpleNamespace(eos_token_id=-1)
        self.ops = SimpleNamespace(argmax=lambda logits: logits.argmax(-1))
        self.max_seq_len = max_seq_len
        self.device = torch.device("cpu")
        self.cuda_graph = True

    def _step(self, start_pos: int, seqlen: int) -> torch.Tensor:
        return torch.tensor([start_pos])

    def _run(self, ids: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot((ids[:, -1] + 1) % _VOCAB, _VOCAB).float() + step


def test_a_decode_step_off_cuda_runs_eagerly(graph_calls):
    """With ``cuda_graph`` on, a model on the CPU decodes without capturing a graph."""
    model = _Model()
    model(torch.ones(1, 3, dtype=torch.long), 0)
    for pos in range(3, 6):
        logits = model(torch.full((1, 1), pos), pos)
        assert logits.argmax().item() == pos + 1 and logits.min().item() == pos
    assert not graph_calls


def test_forward_takes_one_request_in_order():
    """``forward`` refuses a batch, an empty call, a call at ``start_pos > 0`` before a
    prefill or out of order, and a sequence past ``max_seq_len``; a refused call leaves the
    request where it was."""
    model = _Model(max_seq_len=18)
    ids = torch.zeros(1, 19, dtype=torch.long)
    refused = [
        (ids[:, :4].expand(2, -1), 0, "batch of 2"),
        (ids[:, :0], 0, "at least one token"),
        (ids[:, :1], 4, "prefill at start_pos 0"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :10], 0)
    refused = [
        (ids, 0, "max_seq_len"),
        (ids[:, :1], 11, "continues at position 10"),
        (ids[:, :9], 10, "max_seq_len"),
    ]
    for step, pos, match in refused:
        with pytest.raises(ValueError, match=match):
            model(step, pos)
    model(ids[:, :3], 10)
    model(ids[:, :1], 13)


def test_generate_takes_the_argmax_until_eos():
    """``generate`` runs a prefill and one decode step per fed-back token, and stops after an
    EOS token, given as one id, as several, or by ``config.eos_token_id``."""
    model = _Model()
    forward, calls = model.forward, []
    model.forward = lambda *args: calls.append(args) or forward(*args)
    out = model.generate([1, 2, 3], 12, eos_token_id=-1)
    assert out == [4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
    assert len(calls) == 12, "a prefill and one decode step per fed-back token"
    assert model.generate([1, 2, 3], 12, eos_token_id=0) == [4, 5, 6, 7, 0]
    assert model.generate([1, 2, 3], 12, eos_token_id=[-1, 1]) == [4, 5, 6, 7, 0, 1]
    model.config.eos_token_id = 6
    assert model.generate([1, 2, 3], 12) == [4, 5, 6], "config.eos_token_id stops it"


def test_generate_stops_when_the_sequence_fills_max_seq_len():
    """From a three-token prompt with ``max_seq_len`` 8 and no EOS, ``generate`` returns five
    tokens to fill its eight positions, none for ``max_new_tokens`` 0, and refuses a negative one."""
    model = _Model(max_seq_len=8)
    assert len(model.generate([1, 2, 3], 12, eos_token_id=-1)) == 5
    assert model.generate([1, 2, 3], 0) == []
    with pytest.raises(ValueError, match="max_new_tokens"):
        model.generate([1, 2, 3], -1)
