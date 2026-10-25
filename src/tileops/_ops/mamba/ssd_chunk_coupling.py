"""
CB Producer Op - High-level interface for CB matrix computation.
"""

from typing import ClassVar, Mapping

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.mamba.call_spec import SSDChunkCouplingCall, SSDChunkCouplingFwdInterface
from tileops._kernels.mamba.ssd_chunk_coupling import SSDChunkCouplingKernel
from tileops._ops.op_base import Op
from tileops._perf.profile import tensor_core_roof
from tileops.backend import Target

__all__ = ["SSDChunkCouplingFwdOp"]


class SSDChunkCouplingFwdOp(Op):
    """CB (C@B) matrix producer operator.

    Computes cb[b,c,g,l,s] = sum_n C[b,c*Q+l,g,n] * B[b,c*Q+s,g,n]
    with causal masking (cb[l,s] = 0 if s > l).
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_chunk_coupling": SSDChunkCouplingKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "ssd_chunk_coupling": SSDChunkCouplingFwdInterface
    }

    def __init__(
        self,
        chunk_size: int,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_size: Chunk length (Q).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.chunk_size = chunk_size
        super().__init__(target=target)

    def forward(
        self,
        C: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C: [batch, S, G, N]  dtype
            B: [batch, S, G, N]  dtype

        Returns:
            cb: [batch, num_chunks, G, Q, Q]  dtype
        """
        C = C.contiguous()
        B = B.contiguous()
        batch, seq_len, n_groups, d_state = C.shape
        kernel = self.kernel_for(
            "ssd_chunk_coupling",
            lambda: SSDChunkCouplingCall(
                batch=batch,
                seq_len=seq_len,
                n_groups=n_groups,
                d_state=d_state,
                chunk_len=self.chunk_size,
                dtype=C.dtype,
                device=C.device,
            ),
        )
        return kernel(C, B)

    def roof_key(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["C"][1])
