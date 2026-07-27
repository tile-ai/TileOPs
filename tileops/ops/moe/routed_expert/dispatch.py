"""Dispatch adapters that normalize routed rows to a tight ``ExpertBatch``.

The compute contract remains communication independent.  Communication
backends are adapted here and their opaque combine handles never enter the
expert MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor

from .abc import ExpertBatch
from .permute_nopad import MoePermuteNopadFwdOp

__all__ = [
    "DeepEPDispatchAdapter",
    "ExpertDispatchResult",
    "LocalDispatchHandle",
    "LocalExpertDispatcher",
]


class _DeepEPEvent(Protocol):
    def current_stream_wait(self) -> None: ...


class _DeepEPBuffer(Protocol):
    def dispatch(self, x: Tensor, **kwargs: Any) -> tuple[Any, ...]: ...


@dataclass(frozen=True)
class LocalDispatchHandle:
    """Metadata required to invert a local dispatch.

    ``forward_mapping`` maps each flattened source ``(token, top-k slot)`` pair
    to its row in the tight expert-major batch.  Routing weights are deliberately
    kept in :class:`ExpertDispatchResult` and are not applied by dispatch.
    """

    forward_mapping: Tensor
    num_tokens: int
    top_k: int


@dataclass(frozen=True)
class ExpertDispatchResult:
    """Normalized output of a local or external expert dispatch.

    ``routing_weights`` has one entry per physical row in ``batch.hidden``.
    Only rows before ``batch.valid_rows`` are defined.  ``combine_handle`` is
    owned by the dispatch backend and must be passed back to that backend; the
    TileOps expert compute path never inspects it.
    """

    batch: ExpertBatch
    routing_weights: Tensor
    combine_handle: object
    event: object | None = None

    def __post_init__(self) -> None:
        if self.routing_weights.ndim != 1:
            raise ValueError(
                f"routing_weights must be rank 1 [capacity], got {self.routing_weights.shape}"
            )
        if self.routing_weights.shape[0] != self.batch.capacity:
            raise ValueError(
                "routing_weights capacity must equal batch capacity; got "
                f"{self.routing_weights.shape[0]} and {self.batch.capacity}"
            )
        if self.routing_weights.dtype != torch.float32:
            raise ValueError(
                f"routing_weights must use torch.float32, got {self.routing_weights.dtype}"
            )
        if self.routing_weights.device != self.batch.hidden.device:
            raise ValueError("routing_weights and batch.hidden must be on the same device")


class LocalExpertDispatcher:
    """World-size-one reference dispatcher with tight expert-major output."""

    def __init__(
        self,
        num_experts: int,
        *,
        total_tokens: int | None = None,
        top_k: int | None = None,
        hidden_size: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.num_experts = num_experts
        self._permute = MoePermuteNopadFwdOp(
            total_tokens=total_tokens,
            top_k=top_k,
            num_experts=num_experts,
            hidden_size=hidden_size,
            dtype=dtype,
        )

    def dispatch(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor,
        topk_weights: Tensor,
    ) -> ExpertDispatchResult:
        """Expand every routed pair and sort it by local expert.

        Duplicate expert selections are preserved as distinct routed pairs.
        ``topk_ids`` must contain local expert IDs in ``[0, num_experts)``.
        """
        _validate_routing_inputs(hidden_states, topk_ids, topk_weights)
        if topk_ids.dtype != torch.int32:
            raise ValueError(
                f"LocalExpertDispatcher requires topk_ids.dtype torch.int32, got {topk_ids.dtype}"
            )

        permuted, true_offsets, true_sizes, _, forward_mapping = self._permute(
            hidden_states, topk_ids
        )
        expert_offsets = torch.empty(
            self.num_experts + 1, dtype=torch.int32, device=hidden_states.device
        )
        expert_offsets[:-1].copy_(true_offsets)
        expert_offsets[-1:].copy_(true_offsets[-1:] + true_sizes[-1:])

        dispatched_weights = torch.empty(
            topk_ids.numel(), dtype=torch.float32, device=hidden_states.device
        )
        dispatched_weights.scatter_(
            0,
            forward_mapping.to(torch.int64),
            topk_weights.flatten(),
        )
        handle = LocalDispatchHandle(
            forward_mapping=forward_mapping,
            num_tokens=hidden_states.shape[0],
            top_k=topk_ids.shape[1],
        )
        return ExpertDispatchResult(
            batch=ExpertBatch(hidden=permuted, expert_offsets=expert_offsets),
            routing_weights=dispatched_weights,
            combine_handle=handle,
        )


class DeepEPDispatchAdapter:
    """Adapt a DeepEP V2 ``ElasticBuffer`` dispatch to ``ExpertBatch``.

    The adapter intentionally imports no DeepEP package.  The caller owns and
    injects an initialized ``ElasticBuffer``.  Dispatch uses DeepEP's expanded
    layout with ``expert_alignment=1``: one row per routed pair, grouped by
    local expert, without inter-expert padding.  The received activation buffer
    is therefore borrowed directly by ``ExpertBatch`` without a data copy.

    ``do_cpu_sync=False`` keeps the received row count on the GPU.  DeepEP's
    inclusive per-expert prefix sum is converted to TileOps' exclusive
    ``[0, end_0, ..., end_E]`` convention on the current CUDA stream.
    """

    def __init__(
        self,
        buffer: _DeepEPBuffer,
        *,
        num_experts: int,
        num_local_experts: int,
        num_max_tokens_per_rank: int,
        num_sms: int = 0,
        topk_ids_dtype: torch.dtype = torch.int64,
    ) -> None:
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_local_experts <= 0:
            raise ValueError(f"num_local_experts must be positive, got {num_local_experts}")
        if num_local_experts > num_experts:
            raise ValueError(
                "num_local_experts cannot exceed num_experts; got "
                f"{num_local_experts} and {num_experts}"
            )
        if num_max_tokens_per_rank <= 0:
            raise ValueError(
                f"num_max_tokens_per_rank must be positive, got {num_max_tokens_per_rank}"
            )
        if topk_ids_dtype not in (torch.int32, torch.int64):
            raise ValueError(
                f"topk_ids_dtype must be torch.int32 or torch.int64, got {topk_ids_dtype}"
            )
        self.buffer = buffer
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_sms = num_sms
        self.topk_ids_dtype = topk_ids_dtype

    def dispatch(
        self,
        hidden_states: Tensor,
        topk_ids: Tensor | None,
        topk_weights: Tensor,
        *,
        expert_offsets: Tensor | None = None,
        cached_handle: object | None = None,
    ) -> ExpertDispatchResult:
        """Run asynchronous DeepEP dispatch and normalize its GPU metadata.

        ``expert_offsets`` may be supplied by a graph-aware caller to keep its
        address stable across replays.  Waiting uses a CUDA stream dependency;
        it does not synchronize the host.  To reuse a decode layout, pass the
        prior ``combine_handle`` as ``cached_handle`` and set ``topk_ids=None``.
        """
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "DeepEP BF16 dispatch requires hidden_states.dtype "
                f"torch.bfloat16, got {hidden_states.dtype}"
            )
        if cached_handle is None:
            if topk_ids is None:
                raise ValueError("topk_ids is required when cached_handle is not provided")
            _validate_routing_inputs(hidden_states, topk_ids, topk_weights)
            if topk_ids.dtype != self.topk_ids_dtype:
                raise ValueError(
                    "DeepEP topk_ids dtype must match the adapter's configured "
                    f"topk_ids_dtype {self.topk_ids_dtype}, got {topk_ids.dtype}"
                )
        else:
            if topk_ids is not None:
                raise ValueError("topk_ids must be None when cached_handle is provided")
            _validate_cached_routing_inputs(hidden_states, topk_weights, cached_handle)
            if not getattr(cached_handle, "do_expand", False):
                raise ValueError("cached_handle must come from an expanded DeepEP dispatch")
            if getattr(cached_handle, "expert_alignment", None) != 1:
                raise ValueError("cached_handle must use expert_alignment=1")

        recv_x, recv_topk_ids, recv_weights, handle, event = self.buffer.dispatch(
            hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_tokens_per_rank,
            expert_alignment=1,
            num_sms=self.num_sms,
            async_with_compute_stream=True,
            do_cpu_sync=False,
            do_expand=True,
            handle=cached_handle,
        )

        if isinstance(recv_x, tuple):
            raise ValueError("DeepEP returned FP8 data/scales; this M6 adapter supports BF16 only")
        if recv_topk_ids is not None:
            raise RuntimeError("DeepEP expanded dispatch unexpectedly returned recv_topk_idx")
        if recv_weights is None:
            raise RuntimeError("DeepEP dispatch did not return routing weights required by combine")
        if not hasattr(event, "current_stream_wait"):
            raise TypeError("DeepEP dispatch event must provide current_stream_wait()")

        event.current_stream_wait()
        inclusive_offsets = getattr(handle, "psum_num_recv_tokens_per_expert", None)
        if not isinstance(inclusive_offsets, Tensor):
            raise TypeError("DeepEP handle must expose psum_num_recv_tokens_per_expert")
        if inclusive_offsets.shape != (self.num_local_experts,):
            raise ValueError(
                "DeepEP per-expert prefix sum must have shape "
                f"[{self.num_local_experts}], got {inclusive_offsets.shape}"
            )
        if inclusive_offsets.dtype != torch.int32:
            raise ValueError(
                f"DeepEP per-expert prefix sum must use torch.int32, got {inclusive_offsets.dtype}"
            )
        if inclusive_offsets.device != recv_x.device:
            raise ValueError("DeepEP prefix sum and received activations must share a device")

        if expert_offsets is None:
            expert_offsets = torch.empty(
                self.num_local_experts + 1,
                dtype=torch.int32,
                device=recv_x.device,
            )
        else:
            _validate_offsets_buffer(expert_offsets, self.num_local_experts, recv_x.device)
        expert_offsets[0].zero_()
        expert_offsets[1:].copy_(inclusive_offsets)

        return ExpertDispatchResult(
            batch=ExpertBatch(hidden=recv_x, expert_offsets=expert_offsets),
            routing_weights=recv_weights,
            combine_handle=handle,
            event=event,
        )


def _validate_routing_inputs(
    hidden_states: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
) -> None:
    if not hidden_states.is_cuda:
        raise ValueError("hidden_states must be a CUDA tensor")
    if not topk_ids.is_cuda or not topk_weights.is_cuda:
        raise ValueError("topk_ids and topk_weights must be CUDA tensors")
    if not (hidden_states.device == topk_ids.device == topk_weights.device):
        raise ValueError("hidden_states, topk_ids, and topk_weights must share a device")
    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank 2 [T, H], got {hidden_states.shape}")
    if topk_ids.ndim != 2:
        raise ValueError(f"topk_ids must be rank 2 [T, K], got {topk_ids.shape}")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_weights must have the same shape as topk_ids; got "
            f"{topk_weights.shape} and {topk_ids.shape}"
        )
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "routing token count must equal hidden_states token count; got "
            f"{topk_ids.shape[0]} and {hidden_states.shape[0]}"
        )
    if topk_weights.dtype != torch.float32:
        raise ValueError(f"topk_weights must use torch.float32, got {topk_weights.dtype}")


def _validate_cached_routing_inputs(
    hidden_states: Tensor,
    topk_weights: Tensor,
    cached_handle: object,
) -> None:
    cached_topk_ids = getattr(cached_handle, "topk_idx", None)
    if not isinstance(cached_topk_ids, Tensor):
        raise TypeError("cached_handle must expose its device topk_idx tensor")
    _validate_routing_inputs(hidden_states, cached_topk_ids, topk_weights)


def _validate_offsets_buffer(
    expert_offsets: Tensor,
    num_local_experts: int,
    device: torch.device,
) -> None:
    if expert_offsets.shape != (num_local_experts + 1,):
        raise ValueError(
            "expert_offsets buffer must have shape "
            f"[{num_local_experts + 1}], got {expert_offsets.shape}"
        )
    if expert_offsets.dtype != torch.int32:
        raise ValueError(f"expert_offsets buffer must use torch.int32, got {expert_offsets.dtype}")
    if expert_offsets.device != device:
        raise ValueError("expert_offsets buffer and received activations must share a device")
