from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba import SSDChunkScanFwdKernel
from tileops.perf.profile import tensor_core_roof

from ..op_base import Op

__all__ = ["SSDChunkScanFwdOp"]


class SSDChunkScanFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) fused chunk output operator.

    Fuses the history (prev_states) contribution and intra-chunk causal decay
    into a single pass, computing:

      out[l, p] = exp(dA_cumsum[l]) * (C[l] @ prev_states)
                + sum_{s <= l} cb[l, s] * exp(dA_cumsum[l] - dA_cumsum[s]) * dt[s] * x[s, p]

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_chunk_scan_fwd": SSDChunkScanFwdKernel
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
            tune: Whether to autotune the tile config when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, dtype and device."""
        (
            batch,
            num_chunks,
            chunk_len,
            n_heads,
            d_head,
            d_state,
            n_groups,
            dtype,
            _device,
        ) = call
        return call, lambda: self.kernel_map["ssd_chunk_scan_fwd"](
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, tune=self.tune
        )

    def forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused SSD chunk output pass.

        Args:
            x:           (batch, seqlen, n_heads, d_head)                    dtype
            cb:          (batch, num_chunks, n_groups, chunk_len, chunk_len)  dtype
            dA_cumsum:   (batch, n_heads, num_chunks, chunk_len)              float32
            C:           (batch, seqlen, n_groups, d_state)                   dtype
            prev_states: (batch, num_chunks, n_heads, d_head, d_state)        float32
            dt:          (batch, n_heads, num_chunks, chunk_len)              dtype

        Returns:
            y: (batch, seqlen, n_heads, d_head) float32
        """
        return self._call_boundary(x, cb, dA_cumsum, C, prev_states, dt)

    def _eager_forward(
        self,
        x: torch.Tensor,
        cb: torch.Tensor,
        dA_cumsum: torch.Tensor,
        C: torch.Tensor,
        prev_states: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, _seq_len, n_heads, d_head = x.shape
        num_chunks, n_groups, chunk_len = cb.shape[1], cb.shape[2], cb.shape[3]
        d_state = C.shape[3]
        kernel = self.kernel_for(
            "ssd_chunk_scan_fwd",
            (x, cb, dA_cumsum, C, prev_states, dt),
            (
                batch,
                num_chunks,
                chunk_len,
                n_heads,
                d_head,
                d_state,
                n_groups,
                x.dtype,
                x.device.index,
            ),
        )
        return kernel(
            x.contiguous(),
            cb.contiguous(),
            dA_cumsum.contiguous(),
            C.contiguous(),
            prev_states.contiguous(),
            dt.contiguous(),
        )

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["x"][1])
