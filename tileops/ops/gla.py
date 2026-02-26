"""GLA (Gated Linear Attention) Forward Op."""

from typing import Any, Dict, Optional

import torch

from tileops.ops.op import Op


class GLAFwdOp(Op):
    """Op wrapper for GLA (Gated Linear Attention) forward pass.

    Dispatches to GLAFwdKernel which implements the 4-stage chunked forward:
    cumulative gate sum -> inter-chunk recurrence -> intra-chunk attention -> output.

    Args:
        batch: Batch size B.
        seq_len: Sequence length T. Must be divisible by chunk_size.
        heads: Number of query/key/value heads H.
        dim_k: Key/query head dimension K.
        dim_v: Value head dimension V.
        chunk_size: Chunk size BT (default 64).
        scale: Query scale factor. Defaults to 1/sqrt(dim_k).
        output_final_state: If True, also return the final hidden state.
        dtype: Input tensor dtype (default torch.float16).
        tune: Whether to run kernel autotuning.
        kernel_map: Optional override for the kernel dispatch map.

    Example:
        >>> op = GLAFwdOp(batch=2, seq_len=128, heads=8, dim_k=64, dim_v=64)
        >>> q = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.float16)
        >>> k = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.float16)
        >>> v = torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.float16)
        >>> g = torch.nn.functional.logsigmoid(
        ...     torch.randn(2, 128, 8, 64, device='cuda', dtype=torch.float16))
        >>> o, final_state = op(q, k, v, g)
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        chunk_size: int = 64,
        scale: Optional[float] = None,
        output_final_state: bool = False,
        dtype: torch.dtype = torch.float16,
        tune: bool = False,
        kernel_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map', '__class__')}
        # resolve default scale before storing
        if params['scale'] is None:
            params['scale'] = dim_k**-0.5
        for k, v in params.items():
            setattr(self, k, v)
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gla_fwd"](**params)

    @property
    def default_kernel_map(self) -> Dict[str, Any]:
        from tileops.kernels.gla import GLAFwdKernel
        return {"gla_fwd": GLAFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run GLA forward.

        Args:
            q: Queries [B, T, H, K].
            k: Keys [B, T, H, K].
            v: Values [B, T, H, V].
            g: Log-space forget gates [B, T, H, K].
            initial_state: Optional initial hidden state [B, H, K, V] float32.

        Returns:
            Tuple of (o [B, T, H, V], final_state [B, H, K, V] or None).
        """
        return self.kernel(q, k, v, g, initial_state)
