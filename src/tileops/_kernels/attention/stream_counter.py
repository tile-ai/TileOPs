"""Per-stream zeroed counters the persistent attention programs claim work or count on."""

import torch

__all__ = ["stream_counter"]


def stream_counter(counters: dict, device: torch.device, size: int = 2) -> torch.Tensor:
    """A zeroed int32 counter of *size* slots for the current stream, which the launch leaves
    zeroed again. Concurrent launches on one counter would share its counts, so each stream
    keeps its own in *counters*, and each captured graph its own from the graph's pool."""
    if torch.cuda.is_current_stream_capturing():
        return torch.zeros(size, dtype=torch.int32, device=device)
    stream = torch.cuda.current_stream(device).cuda_stream
    counter = counters.get(stream)
    if counter is None:
        counter = counters[stream] = torch.zeros(size, dtype=torch.int32, device=device)
    return counter
