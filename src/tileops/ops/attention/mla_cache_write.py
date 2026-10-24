"""MLA latent cache write operator (host-side Op layer).

Provides:
  - MLAKVCacheWriteFwdOp: kv_cache.view(-1, DC + PE)[slot_mapping] = cat(kv_c, rope(k_pe) or k_pe)
"""

from typing import ClassVar, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention.call_spec import (
    MLAKVCacheWriteCall,
    MLAKVCacheWriteFwdInterface,
)
from tileops.kernels.attention.mla_cache_write import MLAKVCacheWriteKernel
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.op_base import Op

__all__ = ["MLAKVCacheWriteFwdOp"]


class MLAKVCacheWriteFwdOp(Op):
    """Write the MLA latent and rope key of new tokens into one paged latent cache.

    Each token's row is its ``kv_c`` followed by its ``k_pe``, the cache layout of
    ``MLAPagedFwdOp``. The slots come from the caller as one flattened row index per token,
    ``-1`` skipping the token; distinct non-negative slots are the caller's obligation, and
    a call writes nothing outside the rows it names. With ``fuse_rope`` the ``k_pe`` half of
    the row is rotated at the token's position as it is written; ``k_pe`` itself is not
    modified. Each rotated value is computed in float32 and rounded once to the input dtype,
    so it can differ by a step of that dtype from a rotation evaluated in 16-bit arithmetic.
    With an FP8 cache the row is divided by ``scale`` and cast to E4M3FN. NaN is preserved;
    finite overflow and infinities saturate to the largest finite magnitude with the same
    sign.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mla_cache_write_kernel": MLAKVCacheWriteKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mla_cache_write": MLAKVCacheWriteFwdInterface
    }

    def __init__(
        self,
        fuse_rope: bool = False,
        rope_layout: str = "neox",
        *,
        target: Target = None,
    ) -> None:
        """Build the op; the cache geometry and dtypes arrive with each call.

        Args:
            fuse_rope: Rotate ``k_pe`` at ``positions`` from ``cos_sin_cache`` as it is
                written. The call then passes both tensors.
            rope_layout: Which elements of ``k_pe`` rotate together: ``"neox"`` pairs
                element ``i`` with ``i + PE / 2``, ``"interleaved"`` pairs ``2i`` with
                ``2i + 1``.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.fuse_rope = fuse_rope
        self.rope_layout = rope_layout
        super().__init__(target=target)

    def forward(
        self,
        kv_c: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
    ) -> None:
        """Write each token's ``kv_c`` and ``k_pe`` into the cache row its slot names.

        Args:
            kv_c: The latent of the new tokens, ``(tokens, DC)`` in float16 or bfloat16.
            k_pe: Their rope keys, ``(tokens, PE)`` in the same dtype.
            kv_cache: The latent cache, ``(num_pages, page_size, DC + PE)``, written in
                place. Its dtype is the cache dtype: float16 or bfloat16 matching ``kv_c``,
                or float8_e4m3fn.
            slot_mapping: ``int64`` ``(tokens,)``; entry ``n`` is
                ``page * page_size + offset``, the flattened row token ``n`` writes, or
                ``-1`` to skip it.
            scale: ``float32`` ``(1,)`` divisor of the stored values, required exactly when
                the cache is float8_e4m3fn.
            positions: ``int64`` ``(tokens,)``, the ``cos_sin_cache`` row each token
                rotates by; required exactly when ``fuse_rope``.
            cos_sin_cache: ``(max_position, PE)`` in float16, bfloat16 or float32: the
                ``PE / 2`` cosines of a position, then its ``PE / 2`` sines; required
                exactly when ``fuse_rope``.

        Returns:
            None; the cache is mutated and no other tensor is touched.
        """
        kv_c = kv_c.contiguous()
        k_pe = k_pe.contiguous()
        slot_mapping = slot_mapping.contiguous()
        scale = None if scale is None else scale.contiguous()
        positions = None if positions is None else positions.contiguous()
        cos_sin_cache = None if cos_sin_cache is None else cos_sin_cache.contiguous()
        call = MLAKVCacheWriteCall(
            latent_dim=kv_c.shape[1],
            rope_dim=k_pe.shape[1],
            num_pages=kv_cache.shape[0],
            page_size=kv_cache.shape[1],
            dtype=kv_c.dtype,
            cache_dtype=kv_cache.dtype,
            fuse_rope=self.fuse_rope,
            rope_layout=self.rope_layout,
            cos_sin_dtype=None if cos_sin_cache is None else cos_sin_cache.dtype,
            max_position=0 if cos_sin_cache is None else cos_sin_cache.shape[0],
            device=kv_c.device,
        )
        # kv_cache is declared contiguous and mutated, so it is handed over as it arrived:
        # a copy would take the write with it.
        self.kernel_for("mla_cache_write", call)(
            kv_c, k_pe, kv_cache, slot_mapping, scale, positions, cos_sin_cache
        )
        return None
