from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba import SSDChunkStateFwdKernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["SSDChunkStateFwdOp"]


class SSDChunkStateFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) chunk state forward operator.

    Computes the chunk-end State Space Model (SSM) state for each chunk:

      out[b, c, h, p, n] =
          sum_{l=0}^{Q-1}
              x[b, c*Q+l, h, p]
              * B[b, c*Q+l, g(h), n]
              * exp(dA_cumsum[b,h,c,Q-1] - dA_cumsum[b,h,c,l])
              * dt[b, h, c, l]
              * (1 if seq_idx is None else (seq_idx[b,c*Q+Q-1] >= 0 and seq_idx[b,c*Q+l] == seq_idx[b,c*Q+Q-1]))

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_chunk_state_fwd": SSDChunkStateFwdKernel
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune:       Whether to autotune the tile config when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtypes, seq-idx presence and device."""
        (
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            dtype,
            dt_dtype,
            has_seq_idx,
            _device,
        ) = call
        return call, lambda: self.kernel_map["ssd_chunk_state_fwd"](
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            dtype,
            has_seq_idx=has_seq_idx,
            dt_dtype=dt_dtype,
            tune=self.tune,
        )

    def forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the SSD chunk state forward pass.

        Args:
            x:          (batch, seq_len, n_heads, d_head)
            Bmat:       (batch, seq_len, n_groups, d_state)
            dt:         (batch, n_heads, num_chunks, chunk_len), x's dtype or float32
            dA_cumsum:  (batch, n_heads, num_chunks, chunk_len) float32
            seq_idx:    (batch, seq_len) int32, optional

        Returns:
            states: (batch, num_chunks, n_heads, d_head, d_state) float32
        """
        return self._call_boundary(x, Bmat, dt, dA_cumsum, seq_idx)

    def _eager_forward(
        self,
        x: torch.Tensor,
        Bmat: torch.Tensor,
        dt: torch.Tensor,
        dA_cumsum: torch.Tensor,
        seq_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, seq_len, n_heads, d_head = x.shape
        num_chunks, chunk_len = dt.shape[2], dt.shape[3]
        n_groups, d_state = Bmat.shape[2], Bmat.shape[3]
        kernel = self.kernel_for(
            "ssd_chunk_state_fwd",
            (x, Bmat, dt, dA_cumsum, seq_idx),
            (
                batch,
                num_chunks,
                chunk_len,
                n_heads,
                d_head,
                d_state,
                n_groups,
                x.dtype,
                dt.dtype,
                seq_idx is not None,
                x.device.index,
            ),
        )

        x = x.contiguous()
        Bmat = Bmat.contiguous()
        dt = dt.contiguous()
        dA_cumsum = dA_cumsum.contiguous()

        if seq_idx is None:
            # The kernel built for this call has no seq_idx branch, so this
            # buffer only fills the argument slot and is never read.
            seq_idx = x.new_empty(batch, seq_len, dtype=torch.int32)
        else:
            seq_idx = seq_idx.contiguous()

        return kernel(x, Bmat, dt, dA_cumsum, seq_idx)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["x"][1])
