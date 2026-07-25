from typing import Dict, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import SSDStatePassingFwdKernel

from .op_base import Op

__all__ = ["SSDStatePassingFwdOp"]


class SSDStatePassingFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) state passing forward operator.

    Performs the inter-chunk recurrent scan:

      s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

    with s_{-1} = initial_states (or 0 if has_initial_states=False).

    Args:
        batch:              Batch size.
        num_chunks:         Number of chunks (seq_len / chunk_len).
        n_heads:            Number of heads.
        d_state:            State Space Model (SSM) state dimension.
        has_initial_states: Whether to use initial_states as s_{-1}.
        dtype:              Data type for inputs (float16 or bfloat16).
        tune:               Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        has_initial_states: bool = True,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = None
        self.num_chunks = None
        self.n_heads = None
        self.d_state = None
        self.has_initial_states = has_initial_states
        self.dtype = None
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_state_passing_fwd": SSDStatePassingFwdKernel}

    def _get_kernel(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            num_chunks,
            n_heads,
            d_state,
            self.has_initial_states,
            dtype,
            device_index,
            self.tune,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_state_passing_fwd"](
                batch,
                num_chunks,
                n_heads,
                d_state,
                has_initial_states=self.has_initial_states,
                dtype=dtype,
                tune=self.tune,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        initial_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SSD state passing forward pass.

        Args:
            states:           (batch, num_chunks, n_heads, d_state)
            dA_chunk_cumsum:  (batch, n_heads, num_chunks) float32
            initial_states:   (batch, n_heads, d_state) float32

        Returns:
            out:          (batch, num_chunks, n_heads, d_state) float32
            final_states: (batch, n_heads, d_state) float32
        """
        if not states.is_cuda:
            raise ValueError("states must be a CUDA tensor")
        if states.ndim != 4:
            raise ValueError("states must have shape [batch, num_chunks, n_heads, d_state]")
        batch, num_chunks, n_heads, d_state = states.shape
        if dA_chunk_cumsum.shape != (batch, n_heads, num_chunks):
            raise ValueError("dA_chunk_cumsum must have shape [batch, n_heads, num_chunks]")
        if initial_states.shape != (batch, n_heads, d_state):
            raise ValueError("initial_states must have shape [batch, n_heads, d_state]")

        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.dtype = states.dtype
        self.kernel = self._get_kernel(
            batch, num_chunks, n_heads, d_state, states.dtype, states.device.index)

        states = states.contiguous()
        dA_chunk_cumsum = dA_chunk_cumsum.contiguous()
        initial_states = initial_states.contiguous()

        return self.kernel(states, dA_chunk_cumsum, initial_states)
