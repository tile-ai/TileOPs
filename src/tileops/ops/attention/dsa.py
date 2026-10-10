from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import DSADecodeBasicKernel, DSADecodeKernel, DSADecodeWSKernel
from tileops.kernels.attention.call_spec import DSADecodeCall, SparseMLADecodeFwdInterface
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["DSADecodeWithKVCacheFwdOp"]


class DSADecodeWithKVCacheFwdOp(Op):
    """DeepSeek Sparse Attention (DSA) decode.

    This operation is part of a sparse attention mechanism, designed for use in decoding
    with key-value (KV) caching.

    The layout of the operation is BSHD.

    The in-tree kernels serve causal calls with a power-of-two value dimension
    and a zero or power-of-two tail dimension, subject to shared-memory limits.
    """

    # The WGMMA warp-specialized kernels serve SM90 -- the seesaw kernel value dim 512,
    # the older one the other widths it covers; the architecture-agnostic basic kernel
    # serves everywhere else. Selection reads the device when a call arrives.
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "dsa_decode_ws_kernel": DSADecodeWSKernel,
        "dsa_decode_kernel": DSADecodeKernel,
        "dsa_decode_basic_kernel": DSADecodeBasicKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "dsa_decode": SparseMLADecodeFwdInterface
    }

    def roofline_inputs(self) -> "dict[str, int]":
        """The keys this call's selection makes it score, which its flops follow, and the
        distinct ``kv`` rows they reach, which its bytes follow."""
        from tileops.perf.formulas import dsa_distinct_kv_rows, dsa_selected_keys

        return {
            "selected_keys": dsa_selected_keys(self.last_call),
            "distinct_kv_rows": dsa_distinct_kv_rows(self.last_call),
        }

    def __init__(
        self,
        dim_tail: int,
        stride_kv: int,
        q_start_index_s: int,
        sm_scale: Optional[float] = None,
        is_causal: bool = True,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            dim_tail (int): The dimension of the tail portion of the attention vectors.
            stride_kv (int): The stride for the key-value sequence.
            q_start_index_s (int): The start index for queries in the sequence.
            sm_scale (Optional[float], default=None): Scaling factor for the softmax function.
            is_causal (bool, default=True): Whether the attention is causal
                        (True for causal, False for non-causal).
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.dim_tail = dim_tail
        self.stride_kv = stride_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self._cp0 = cp0
        super().__init__(target=target)

    def _dsa_decode_call(
        self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
    ) -> DSADecodeCall:
        """State what one call is, for selection to filter against."""
        batch, seq_len, heads, q_dim = q.shape
        _, seq_len_kv, heads_kv, _ = kv.shape
        return DSADecodeCall(
            batch=batch,
            seq_len=seq_len,
            seq_len_kv=seq_len_kv,
            heads=heads,
            dim=q_dim - self.dim_tail,
            tail_dim=self.dim_tail,
            dtype=q.dtype,
            topk=indices.shape[3],
            kv_stride=self.stride_kv,
            q_start_index_s=self.q_start_index_s,
            kv_group=heads_kv,
            sm_scale=self.sm_scale,
            is_causal=self.is_causal,
            cp0=self._cp0,
            device=q.device,
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the sparse attention operation.

        Args:
            q (torch.Tensor): The query tensor with shape
                        (batch, seq_len, heads, dim + dim_tail).
            kv (torch.Tensor): The key-value tensor with shape
                        (batch, seq_len_kv, heads_kv, dim + dim_tail).
            indices (torch.Tensor): Indices tensor for sparse attention.

        Returns:
            torch.Tensor: The result of applying the sparse attention
                            operation on the input tensors.
        """
        inputs = (q, kv, indices)
        kernel = self.kernel_for("dsa_decode", self._dsa_decode_call(q, kv, indices))
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])
