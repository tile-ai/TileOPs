"""``CausalLM`` on a model whose forward body is plain tensor arithmetic."""

import pytest
import torch

from models.causal_lm import CausalLM

pytestmark = pytest.mark.nightly


class _Model(CausalLM):
    """Logits of the last token: its id times ``1..8``, plus the call's start position."""

    def __init__(self, device: torch.device) -> None:
        self.max_seq_len = 16
        self.device = device
        self.cuda_graph = True
        self.weights = torch.arange(1, 9, dtype=torch.float32, device=device)

    def _step(self, start_pos: int, seqlen: int) -> torch.Tensor:
        return torch.tensor([start_pos], device=self.device)

    def _run(self, ids: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return ids[:, -1:].float() * self.weights + step


def test_a_decode_step_off_cuda_runs_eagerly(graph_calls):
    """With ``cuda_graph`` on, a model on the CPU decodes without capturing a graph."""
    model = _Model(torch.device("cpu"))
    model(torch.ones(1, 3, dtype=torch.long), 0)
    for pos in range(3, 6):
        logits = model(torch.full((1, 1), pos), pos)
        assert torch.equal(logits, pos * model.weights.view(1, -1) + pos)
    assert not graph_calls
