from typing import ClassVar, Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import SparseMlaBasicKernel, SparseMlaCall, SparseMlaKernel
from tileops.kernels.kernel_base import Kernel
from tileops.perf.profile import tensor_core_roof

from .._compile_boundary_codegen import OperatorSpec
from ..op_base import Op

__all__ = ["DeepSeekSparseAttentionDecodeWithKVCacheFwdOp"]


class DeepSeekSparseAttentionDecodeWithKVCacheFwdOp(Op):
    """
    Sparse Attention Decode Operation with Key-Value Cache for DeepSeek.

    This operation is part of a sparse attention mechanism, designed for use in decoding
    with key-value (KV) caching.

    The layout of the operation is BSHD.

    """

    compile_boundary: ClassVar[tuple[OperatorSpec, ...]] = (OperatorSpec(),)

    def __init__(
        self,
        dim_tail: int,
        stride_kv: int,
        q_start_index_s: int,
        sm_scale: Optional[float] = None,
        is_causal: bool = True,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            kernel_map (Optional[Dict[str, Kernel]], default=None):
                        Optional mapping for custom kernels.
            tune (bool, default=False): Whether to enable kernel tuning.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.target = target
        self.dim_tail = dim_tail
        self.stride_kv = stride_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal

        if q_start_index_s != 0 and q_start_index_s <= stride_kv:
            raise ValueError(
                f"Invalid q_start_index_s={q_start_index_s}:"
                f"must be > stride_kv={stride_kv}. "
                "This indicates incorrect cp0 masking."
                "Ensure queries with pos < stride_kv are masked "
                "to avoid NaNs in early outputs."
            )

        cp0 = q_start_index_s == 0
        self.q_start_index_s = q_start_index_s

        self._cp0 = cp0
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def _sparse_mla_call(
        self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
    ) -> SparseMlaCall:
        """State what one call is, for selection to filter against."""
        batch, seq_len, heads, q_dim = q.shape
        _, seq_len_kv, heads_kv, _ = kv.shape
        return SparseMlaCall(
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
            tune=self.tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        """
        Provides the default kernel map for the operation.

        Returns:
            Dict[str, Kernel]: A dictionary mapping kernel names to kernel functions.
            The WGMMA warp-specialized SparseMlaKernel serves SM90; the
            architecture-agnostic SparseMlaBasicKernel (plain T.gemm) serves
            everywhere else. Selection reads the device when a call arrives.
        """
        return {
            "sparse_mla_kernel": SparseMlaKernel,
            "sparse_mla_basic_kernel": SparseMlaBasicKernel,
        }

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        kv_shape: tuple[int, ...],
        indices_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``shape_rules``: ``o`` drops the tail dims ``q`` carries."""
        return {"o": tuple(q_shape[:-1]) + (q_shape[-1] - self.dim_tail,)}

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
        return self._wrapped(q, kv, indices, self._instance_key)

    def _eager_forward(
        self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        self._validate_dtypes(q, kv, indices)
        self.dtype = q.dtype
        # The legacy roofline formula reads these off the op.
        self.batch, self.seq_len, self.heads, q_dim = q.shape
        _, self.seq_len_kv, self.heads_kv, _ = kv.shape
        self.dim = q_dim - self.dim_tail
        self.topk = indices.shape[3]
        inputs = (q, kv, indices)
        kernel = self.kernel_for("sparse_mla", inputs, self._sparse_mla_call(q, kv, indices))
        return kernel(*inputs)

    def compute_roof(self) -> str:
        """FLOPs are matmul contractions; priced on tensor cores."""
        return tensor_core_roof(self.dtype)
