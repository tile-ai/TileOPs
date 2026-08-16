"""MoE permute op (no-pad variant): counting sort + tight gather without block_m padding.

``expert_map`` is optional. Supplied, this rank owns ``num_experts_local`` of the
``num_experts`` global experts and the scan kernel reads the map to keep only its
own rows; absent, every expert is local and the scan kernel takes no map at all.
The count is a constructor parameter because it sizes the scan kernel's shared
counters and its four per-expert outputs.
"""

import weakref
from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.permute_nopad import MoePermuteNopadKernel

from ...compile_boundary import get_instance
from ...op_base import Op

__all__ = ["MoePermuteNopadFwdOp"]

_HIDDEN_DTYPES = (torch.float16, torch.bfloat16)


class MoePermuteNopadFwdOp(Op):
    """Route tokens to tight (non-padded) expert-contiguous layout.

    The output perm_h has exactly T*K rows with no inter-expert padding,
    enabling smaller intermediate tensors throughout the MoE pipeline.

    Args:
        num_experts: Total number of experts E in the routing table.
        num_experts_local: Number of those experts this rank lays out. Equal to
            ``num_experts`` outside expert parallelism.
        total_tokens: Optional committed number of input tokens T. Preferred
            API infers it from ``hidden_states.shape[0]``.
        top_k: Optional committed number of experts selected per token K.
            Preferred API infers it from ``topk_ids.shape[1]``.
        hidden_size: Optional committed hidden dimension H. Preferred API
            infers it from ``hidden_states.shape[1]``.
        kernel_map: Optional kernel override dict.

    Example:
        >>> op = MoePermuteNopadFwdOp(num_experts=8, num_experts_local=8)
        >>> perm_h, offsets, sizes, expert_offset, fwd_idx = op(hidden_states, topk_ids)
    """

    #: The operator this op registers; a test asserts the graph holds nothing else.
    compile_op_names: ClassVar[Tuple[str, ...]] = ("top::moe_permute_nopad_fwd",)

    def __init__(
        self,
        num_experts: int,
        num_experts_local: int,
        total_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        hidden_size: Optional[int] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        if not 0 < num_experts_local <= num_experts:
            raise ValueError(
                f"num_experts_local must be in (0, {num_experts}], "
                f"got {num_experts_local}"
            )
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_experts_local = num_experts_local
        self.hidden_size = hidden_size
        self._committed_total_tokens = total_tokens
        self._committed_top_k = top_k
        self._committed_hidden_size = hidden_size
        # Identity and in-place version of the map the density check last passed
        # on. A weak reference, so a freed tensor cannot pass as this one through
        # a reused address.
        self._checked_map: Optional[tuple["weakref.ref", int]] = None
        # Whether the last call supplied a map, for eval_roofline().
        self.used_expert_map = False

        self.dispatch_kernel(kernel_map)

    def _validate_dtypes(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor] = None,
    ) -> None:
        if hidden_states.dtype not in _HIDDEN_DTYPES:
            raise ValueError(
                "Expected hidden_states.dtype to be torch.float16 or "
                f"torch.bfloat16, got {hidden_states.dtype}"
            )
        if topk_ids.dtype != torch.int32:
            raise ValueError(
                f"Expected topk_ids.dtype torch.int32, got {topk_ids.dtype}")
        if expert_map is not None and expert_map.dtype != torch.int32:
            raise ValueError(
                f"Expected expert_map.dtype torch.int32, got {expert_map.dtype}")

    def _infer_output_shapes(
        self,
        hidden_states_shape: Tuple[int, ...],
        topk_ids_shape: Tuple[int, ...],
        expert_map_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules`` for the five outputs.

        Every size follows from the two routing shapes and
        ``num_experts_local``; the map's contents are never read here.
        """
        numel = hidden_states_shape[0] * topk_ids_shape[1]
        e_local = self.num_experts_local
        return {
            "perm_h": (numel, hidden_states_shape[1]),
            "true_offsets": (e_local,),
            "true_sizes": (e_local,),
            "expert_first_token_offset": (e_local + 1,),
            "fwd_idx": (numel,),
        }

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import moe_permute_nopad_fwd_roofline
        return moe_permute_nopad_fwd_roofline(self)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_nopad_kernel": MoePermuteNopadKernel}

    def _get_kernel(
        self,
        inputs: Tuple[torch.Tensor, ...],
        total_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype,
        device_index: int | None,
        num_experts_local: Optional[int],
    ) -> Kernel:
        key = (
            total_tokens, top_k, self.num_experts, hidden_size, dtype, device_index,
            num_experts_local,
        )
        return self.get_or_build_kernel(
            "permute_nopad_kernel", inputs,
            key=key,
            build=lambda: self.kernel_map["permute_nopad_kernel"](
                total_tokens, top_k, self.num_experts, hidden_size, dtype,
                num_experts_local=num_experts_local,
            ),
        )

    def _validate_routing_shapes(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor,
    ) -> Tuple[int, int, int]:
        """Check devices and shapes of the two routing tensors; return ``(T, K, H)``."""
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be a CUDA tensor")
        if not topk_ids.is_cuda:
            raise ValueError("topk_ids must be a CUDA tensor")
        if hidden_states.device != topk_ids.device:
            raise ValueError(
                f"Expected hidden_states and topk_ids to be on the same device, "
                f"got {hidden_states.device} and {topk_ids.device}"
            )
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expected hidden_states to be 2D [T, H], got {hidden_states.ndim}D"
            )
        if topk_ids.ndim != 2:
            raise ValueError(
                f"Expected topk_ids to be 2D [T, K], got {topk_ids.ndim}D"
            )
        total_tokens, hidden_size = hidden_states.shape
        topk_tokens, top_k = topk_ids.shape
        if topk_tokens != total_tokens:
            raise ValueError(
                f"Expected topk_ids.shape[0] == hidden_states.shape[0] "
                f"({total_tokens}), got {topk_tokens}"
            )
        if (
            self._committed_total_tokens is not None
            and total_tokens != self._committed_total_tokens
        ):
            raise ValueError(
                f"Expected total_tokens={self._committed_total_tokens}, got {total_tokens}"
            )
        if self._committed_top_k is not None and top_k != self._committed_top_k:
            raise ValueError(f"Expected top_k={self._committed_top_k}, got {top_k}")
        if (
            self._committed_hidden_size is not None
            and hidden_size != self._committed_hidden_size
        ):
            raise ValueError(
                f"Expected hidden_size={self._committed_hidden_size}, got {hidden_size}"
            )
        return total_tokens, top_k, hidden_size

    def _validate_expert_map(self, expert_map: torch.Tensor) -> None:
        """Reject a map that is not a bijection onto the local expert ids.

        The kernel indexes ``num_experts_local`` counters with the map's values, so
        the non-negative entries must be exactly ``0 .. num_experts_local - 1``, each
        once. Reading them costs a host sync, so the verdict is remembered against
        the map's identity and in-place version and re-taken only when either
        changes — a warm-up call pays it, a captured replay does not.
        """
        if tuple(expert_map.shape) != (self.num_experts,):
            raise ValueError(
                f"Expected expert_map.shape ({self.num_experts},), "
                f"got {tuple(expert_map.shape)}")
        if not expert_map.is_cuda:
            raise ValueError("expert_map must be a CUDA tensor")

        checked = self._checked_map
        if (
            checked is not None
            and checked[0]() is expert_map
            and checked[1] == expert_map._version
        ):
            return

        local = sorted(expert_map[expert_map >= 0].tolist())
        if local != list(range(self.num_experts_local)):
            raise ValueError(
                "expert_map must assign the local expert ids "
                f"0 .. {self.num_experts_local - 1} exactly once each; this op was "
                f"built for num_experts_local={self.num_experts_local} and the map's "
                f"non-negative entries are {local}. A gap or a repeat sizes the "
                "kernel's counters for one set of experts while the kernel is handed "
                "another, and the tokens routed past the end are dropped."
            )
        self._checked_map = (weakref.ref(expert_map), expert_map._version)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute without padding.

        Args:
            hidden_states: [T, H] input activations (bf16/fp16).
            topk_ids: [T, K] int32 expert assignments (global ids).
            expert_map: [E] int32 global-to-local expert ids under expert
                parallelism; -1 marks an expert another rank owns. Omit it when
                this rank owns every expert.

        Returns:
            perm_h:                    [T*K, H] tight hidden states; with a map,
                                       only the first M_local rows carry data
            true_offsets:              [E_local] int32 tight start per local expert
            true_sizes:                [E_local] int32 true token count per local expert
            expert_first_token_offset: [E_local+1] int64 non-padded prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx -> tight
                                       slot, -1 for a non-local pair
        """
        return _moe_permute_nopad_fwd(
            hidden_states, topk_ids, expert_map, self._instance_key)

    def _eager_forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate, resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo
        cannot follow.
        """
        total_tokens, top_k, hidden_size = self._validate_routing_shapes(
            hidden_states, topk_ids)
        self._validate_dtypes(hidden_states, topk_ids, expert_map)
        if expert_map is None:
            # A map is the only thing that can name which experts are local, so
            # without one the op must own the whole table.
            if self.num_experts_local != self.num_experts:
                raise ValueError(
                    f"this op was built for num_experts_local="
                    f"{self.num_experts_local} of {self.num_experts} experts and "
                    "needs an expert_map to say which ones they are"
                )
        else:
            if expert_map.device != hidden_states.device:
                raise ValueError(
                    f"Expected expert_map on {hidden_states.device}, "
                    f"got {expert_map.device}")
            self._validate_expert_map(expert_map)

        self.dtype = hidden_states.dtype
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.hidden_states_shape = tuple(hidden_states.shape)
        self.topk_ids_shape = tuple(topk_ids.shape)
        self.used_expert_map = expert_map is not None

        # A map-free kernel and a map-reading one are two different scans, so the
        # argument's presence — not the local expert count — picks between them.
        kernel = self._get_kernel(
            (hidden_states, topk_ids) if expert_map is None
            else (hidden_states, topk_ids, expert_map),
            total_tokens, top_k, hidden_size, hidden_states.dtype,
            hidden_states.device.index,
            self.num_experts_local if expert_map is not None else None,
        )
        if expert_map is None:
            return kernel(hidden_states, topk_ids)
        return kernel(hidden_states, topk_ids, expert_map)


@torch.library.custom_op("top::moe_permute_nopad_fwd", mutates_args=())
def _moe_permute_nopad_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_instance(instance_key)._eager_forward(
        hidden_states, topk_ids, expert_map)


@_moe_permute_nopad_fwd.register_fake
def _moe_permute_nopad_fwd_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(hidden_states.shape), tuple(topk_ids.shape),
        None if expert_map is None else tuple(expert_map.shape),
    )
    device = hidden_states.device
    # Manifest dtypes: ``same_as(hidden_states)`` for the gathered rows, int32 for
    # the index arrays, int64 for the prefix sum.
    return (
        hidden_states.new_empty(shapes["perm_h"]),
        torch.empty(shapes["true_offsets"], dtype=torch.int32, device=device),
        torch.empty(shapes["true_sizes"], dtype=torch.int32, device=device),
        torch.empty(shapes["expert_first_token_offset"], dtype=torch.int64, device=device),
        torch.empty(shapes["fwd_idx"], dtype=torch.int32, device=device),
    )
