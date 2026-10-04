from typing import Callable, ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.attention import (
    GQAPrefillVarlenFwdKernel,
    GQAPrefillVarlenWSFwdKernel,
    GQASlidingWindowVarlenFwdWGMMAPipelinedKernel,
    GQAVarlenFP8FwdKernel,
    GQAVarlenFP8WSFwdKernel,
)
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQAVarlenFwdInterface,
)
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.ops.attention.gqa.parameters import _rope_rotary_dim, _score_softcap
from tileops.ops.op_base import Op
from tileops.perf.profile import tensor_core_roof

__all__ = ["GQAVarlenFwdOp"]


class GQAVarlenFwdOp(Op):
    """Grouped-query attention over packed THD tensors.

    ``cu_seqlens_q`` and ``cu_seqlens_kv`` delimit each request. The interface
    covers both prefill and decode; tensor geometry and sequence metadata come
    from each call, while mask, score, out_dtype, and RoPE semantics are fixed at
    construction. The BUILTIN path implements 16-bit regular and sliding-window
    attention, with or without fused RoPE.

    ``float8_e4m3fn`` Q/K/V are dequantized by one ``q_scale``, ``k_scale`` and
    ``v_scale`` per request and KV head, and name a 16-bit ``out_dtype``; the FP8
    result carries the error of a single e4m3 rounding of the softmax weights, so it
    agrees with the 16-bit path to about 2% relative.

    By default the op does not check the contents of ``cu_seqlens_q`` and
    ``cu_seqlens_kv``. The kernels read and write the packed tensors at the positions the
    offsets name, so offsets that do not start at 0, end at the packed length, and never
    decrease access device memory out of bounds. The caller guarantees well-formed
    offsets, or passes ``validate_inputs=True``, which checks them on every call at the
    cost of a device synchronization and cannot run inside CUDA Graph capture.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gqa_varlen": GQAPrefillVarlenFwdKernel,
        "gqa_varlen_ws": GQAPrefillVarlenWSFwdKernel,
        "gqa_varlen_sliding_window": GQASlidingWindowVarlenFwdWGMMAPipelinedKernel,
        "gqa_varlen_fp8": GQAVarlenFP8FwdKernel,
        "gqa_varlen_fp8_ws": GQAVarlenFP8WSFwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "gqa_varlen": GQAVarlenFwdInterface
    }

    def __init__(
        self,
        is_causal: bool = True,
        window_size_left: int = -1,
        window_size_right: int = -1,
        sm_scale: Optional[float] = None,
        softcap: Optional[float] = None,
        out_dtype: Optional[torch.dtype] = None,
        pos_encoding_mode: str = "none",
        rotary_dim: Optional[int] = None,
        rope_layout: str = "neox",
        validate_inputs: bool = False,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Configure packed variable-length GQA semantics.

        Args:
            is_causal: Apply a bottom-right-aligned causal mask per request.
            window_size_left: Visible keys to the left; ``-1`` is unlimited.
            window_size_right: Visible keys to the right; ``-1`` is unlimited.
            sm_scale: Score scale, or ``None`` for ``1 / sqrt(head_dim)``.
            softcap: Positive score cap; ``None`` or zero disables it.
            out_dtype: Output dtype, inferred from the input when omitted.
            pos_encoding_mode: ``"none"`` or ``"rope"``.
            rotary_dim: Even rotated width; ``None`` uses the full head dimension.
            rope_layout: ``"neox"`` or ``"interleaved"``.
            validate_inputs: Check cumulative offsets against packed tensors on the CPU.
            target: Backend target, or ``None`` to resolve from the input device.
            kernel_map: Optional in-tree kernel overrides.
            tune: Autotune a kernel when it is first built.
        """
        self.is_causal = is_causal
        self.sm_scale = sm_scale
        self.softcap = _score_softcap(softcap)
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.out_dtype = out_dtype
        self.pos_encoding_mode = pos_encoding_mode
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        self.validate_inputs = validate_inputs
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def compute_roof(self) -> str:
        """Varlen attention's contractions are priced on tensor cores."""
        return tensor_core_roof(self.last_call.tensors["q"][1])

    def varlen_call(self, inputs: tuple[Optional[torch.Tensor], ...]) -> AttentionCall:
        """Describe one packed call using tensor shapes and Op semantics."""
        q, k, _v, cu_q, _cu_kv, _qs, _ks, _vs, rope_cos, _rope_sin = inputs
        assert q is not None and k is not None and cu_q is not None
        _, heads, dim = q.shape
        _, heads_kv, _ = k.shape
        return AttentionCall(
            dtype=self.out_dtype or q.dtype,
            batch=cu_q.shape[0] - 1,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=self.is_causal,
            sm_scale=self.sm_scale,
            softcap=self.softcap,
            window_size_left=self.window_size_left,
            window_size_right=self.window_size_right,
            is_fp8=q.dtype == torch.float8_e4m3fn,
            is_uniform=False,
            empty_kv=k.shape[0] == 0,
            fuse_rope=self.pos_encoding_mode == "rope",
            max_position=rope_cos.shape[0] if rope_cos is not None else 1,
            rotary_dim=_rope_rotary_dim(dim, self.rotary_dim)
            if self.pos_encoding_mode == "rope"
            else 0,
            rope_layout=self.rope_layout,
            device=q.device,
        )

    def _get_kernel(
        self, inputs: tuple[Optional[torch.Tensor], ...]
    ) -> Callable[..., torch.Tensor]:
        """Resolve the implementation stored in the Op's single cache layer."""
        return self.kernel_for("gqa_varlen", self.varlen_call(inputs))

    def _check_offsets(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
    ) -> None:
        """Check on the CPU that the offsets span the packed tensors and never decrease."""
        for name, offsets, total in (
            ("cu_seqlens_q", cu_seqlens_q, q.shape[0]),
            ("cu_seqlens_kv", cu_seqlens_kv, k.shape[0]),
        ):
            bounds = [int(value) for value in offsets.detach().cpu().tolist()]
            if bounds[0] != 0:
                raise ValueError(f"{name}[0] must equal 0")
            if bounds[-1] != total:
                raise ValueError(f"{name}[-1] must equal {total}")
            if any(end < start for start, end in zip(bounds[:-1], bounds[1:], strict=True)):
                raise ValueError(f"{name} must be non-decreasing")

    @staticmethod
    def _canonicalize_inputs(
        *inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], ...]:
        return tuple(tensor.contiguous() if tensor is not None else None for tensor in inputs)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run packed Varlen GQA; Q/K/V use ``[total_tokens, heads, dim]``."""
        return self._call_boundary(
            q, k, v, cu_seqlens_q, cu_seqlens_kv, q_scale, k_scale, v_scale, rope_cos, rope_sin
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the implementation and launch it."""
        if self.validate_inputs:
            self._check_offsets(q, k, cu_seqlens_q, cu_seqlens_kv)
        inputs = self._canonicalize_inputs(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            q_scale,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
        )
        return self._get_kernel(inputs)(*inputs)
