"""Call workloads of the fft family."""

import torch

from workloads.fft import FFTWorkload


def _fft(call) -> FFTWorkload:
    shape, dtype = call.tensors["input"]
    return FFTWorkload(shape[-1], getattr(torch, dtype), batch_shape=shape[:-1])


WORKLOADS = {
    "FFTC2CFwdOp": _fft,
}
