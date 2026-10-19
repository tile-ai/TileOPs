"""Per-head q/k RMSNorm fused with RoPE on a packed projection (host-side Op layer).

Provides:
  - FusedQKNormRoPEFwdOp: RMSNorm of each q and k head, then RoPE at each token's position,
    in place
"""

from typing import ClassVar, Mapping

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import FusedQKNormRoPEKernel
from tileops.kernels.norm.call_spec import FusedQKNormRoPECall, FusedQKNormRoPEFwdInterface
from tileops.ops.op_base import Op

__all__ = ["FusedQKNormRoPEFwdOp"]


class FusedQKNormRoPEFwdOp(Op):
    """RMSNorm of every q and k head of a packed projection, then RoPE, in place.

    ``qkv`` holds ``num_heads`` q heads, then ``num_kv_heads`` k heads, then as many v heads,
    ``D`` values each. For token ``n`` and a q or k head with values ``x``:

    $$
    y = \\frac{x}{\\sqrt{\\mathrm{mean}(x^2) + \\epsilon}} \\cdot w
    $$

    with ``w`` the head's ``q_weight`` or ``k_weight``. The first ``R`` values of ``y`` then
    rotate by the angles of row ``positions[n]`` of ``cos_sin_cache``, whose first ``R / 2``
    columns are cosines and last ``R / 2`` sines. ``rope_layout="neox"`` rotates the pairs
    ``(y[i], y[i + R/2])``, ``"interleaved"`` the pairs ``(y[2i], y[2i + 1])``; values past
    ``R`` pass through normalized. The v heads are not touched.

    The arithmetic is float32 and each value rounds once to the ``qkv`` dtype. The
    reciprocal square root is the hardware's approximate one, within two float32 units in
    the last place.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fused_qk_norm_rope": FusedQKNormRoPEKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fused_qk_norm_rope": FusedQKNormRoPEFwdInterface
    }

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        eps: float = 1e-6,
        rope_layout: str = "neox",
        *,
        target: Target = None,
    ) -> None:
        """Build the op. The head dim, rotary width, token count and dtypes come with each call.

        Args:
            num_heads: q heads in ``qkv``.
            num_kv_heads: k heads in ``qkv``, and as many v heads after them.
            eps: Added to the mean square before the root.
            rope_layout: ``"neox"`` to rotate the halves of the rotary values against each
                other, ``"interleaved"`` to rotate adjacent pairs.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps
        self.rope_layout = rope_layout
        super().__init__(target=target)

    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Normalize and rotate the q and k heads of ``qkv`` in place.

        Args:
            qkv: ``(N, (num_heads + 2 * num_kv_heads) * D)``, float16 or bfloat16, contiguous;
                its q and k heads are overwritten.
            q_weight: ``(D,)``, the q heads' RMSNorm weight, in ``qkv``'s dtype.
            k_weight: ``(D,)``, the k heads' RMSNorm weight, in ``qkv``'s dtype.
            cos_sin_cache: ``(P, R)``, float16, bfloat16 or float32: per position, ``R / 2``
                cosines then ``R / 2`` sines.
            positions: int64 ``(N,)``, each token's row of ``cos_sin_cache``.

        Returns:
            None; ``qkv`` is mutated and no other tensor is touched.
        """
        call = FusedQKNormRoPECall(
            heads=self.num_heads,
            kv_heads=self.num_kv_heads,
            dim=q_weight.shape[0],
            rot=cos_sin_cache.shape[1],
            interleaved=self.rope_layout == "interleaved",
            eps=self.eps,
            dtype=qkv.dtype,
            cache_dtype=cos_sin_cache.dtype,
            device=qkv.device,
        )
        # qkv is declared contiguous and mutated, so it is handed over as it arrived: a copy
        # would take the write with it.
        self.kernel_for("fused_qk_norm_rope", call)(
            qkv,
            q_weight.contiguous(),
            k_weight.contiguous(),
            cos_sin_cache.contiguous(),
            positions.contiguous(),
        )
        return None
