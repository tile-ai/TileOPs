from typing import Dict, Optional

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.mamba import SsdStatePassingFwdKernel

from .op import Op

__all__ = ["SsdStatePassingFwdOp"]


class SsdStatePassingFwdOp(Op):
    """Mamba-2 SSD state passing forward operator.

    Performs the inter-chunk recurrent scan:

      s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

    with s_{-1} = initial_states (or 0 if has_initial_states=False).

    Args:
        batch:              Batch size.
        num_chunks:         Number of chunks (seq_len / chunk_len).
        n_heads:            Number of heads.
        d_state:            SSM state dimension.
        has_initial_states: Whether to use initial_states as s_{-1}.
        dtype:              Data type for inputs (float16 or bfloat16).
        tune:               Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_heads: int,
        d_state: int,
        has_initial_states: bool = True,
        dtype: torch.dtype = torch.float16,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_heads = n_heads
        self.d_state = d_state
        self.has_initial_states = has_initial_states
        self.dtype = dtype
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["ssd_state_passing_fwd"](
            batch, num_chunks, n_heads, d_state,
            has_initial_states=has_initial_states, dtype=dtype, tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_state_passing_fwd": SsdStatePassingFwdKernel}

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
        if states.dtype != self.dtype:
            raise ValueError(f"Expected dtype {self.dtype}, got {states.dtype}")

        states = states.contiguous()
        dA_chunk_cumsum = dA_chunk_cumsum.contiguous()
        initial_states = initial_states.contiguous()

        return self.kernel(states, dA_chunk_cumsum, initial_states)
