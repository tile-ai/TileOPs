"""
CB Producer Op - High-level interface for CB matrix computation.
"""

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.mamba.call_spec import SSDChunkCouplingCall, SSDChunkCouplingFwdInterface
from tileops.kernels.mamba.ssd_chunk_coupling import SSDChunkCouplingKernel
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

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
        chunk_len: int,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            chunk_len: Chunk length (Q).
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional pre-initialized kernels
            tune: Whether to autotune
        """
        self.chunk_len = chunk_len
        self.tune = tune
        self.target = target
        self.dispatch_kernel(kernel_map)

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, S, G, N]  dtype
            B_mat: [B, S, G, N]  dtype

        Returns:
            cb: [B, C, G, Q, Q]  dtype
        """
        return self._call_boundary(C_mat, B_mat)

    def _eager_forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        C_mat = C_mat.contiguous()
        B_mat = B_mat.contiguous()
        batch, seq_len, n_groups, d_state = C_mat.shape
        call = SSDChunkCouplingCall(
            batch=batch,
            seq_len=seq_len,
            n_groups=n_groups,
            d_state=d_state,
            chunk_len=self.chunk_len,
            dtype=C_mat.dtype,
            device=C_mat.device,
        )
        kernel = self.kernel_for("ssd_chunk_coupling", call)
        return kernel(C_mat, B_mat)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["C_mat"][1])
