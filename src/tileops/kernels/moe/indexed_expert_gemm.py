"""Route-indexed small-M expert MLP kernels."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops.kernels.kernel_base import Kernel

__all__ = ["IndexedExpertGemmTemplate"]


def _route_metadata_size(num_experts: int, num_routes: int, group_capacity: int) -> int:
    return num_experts + num_routes + num_experts * group_capacity


def _group_capacity(num_tokens: int) -> int:
    return min(num_tokens, 16)


@functools.lru_cache(maxsize=32)
def _route_stats(num_tokens: int, top_k: int, num_experts: int, group_capacity: int):
    num_routes = num_tokens * top_k
    rank_base = num_experts
    list_base = num_experts + num_routes

    @tilelang.jit(out_idx=[])
    def _kernel(threads: int):
        @T.prim_func
        def main(
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            metadata: T.Tensor(
                (_route_metadata_size(num_experts, num_routes, group_capacity),), "int32"
            ),
        ):
            with T.Kernel(1, threads=threads):
                for expert in T.Parallel(num_experts):
                    metadata[expert] = 0
                T.sync_threads()
                for route in T.Parallel(num_routes):
                    expert = expert_ids[route // top_k, route % top_k]
                    rank = T.atomic_add(metadata[expert], 1, return_prev=True)
                    metadata[rank_base + route] = rank
                    if rank < group_capacity:
                        metadata[list_base + expert * group_capacity + rank] = route

        return main

    return _kernel


@functools.lru_cache(maxsize=64)
def _indexed_expert_gemm(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    n: int,
    k: int,
    dtype: str,
    activation: str,
    route_input: bool,
    dispatch_mode: str,
    group_capacity: int,
):
    num_routes = num_tokens * top_k
    gated = activation != "none"
    weight_n = 2 * n if gated else n
    a_shape = (num_tokens, top_k, k) if route_input else (num_tokens, k)
    metadata_size = (
        _route_metadata_size(num_experts, num_routes, group_capacity)
        if dispatch_mode == "grouped"
        else 1
    )
    rank_base = num_experts
    list_base = num_experts + num_routes

    @tilelang.jit(
        out_idx=[],
        compile_flags=["-O3", "-DENABLE_BF16"],
        # The A tile is rewritten inside the K loop, so producer/consumer WS would race.
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def _kernel(block_n: int, block_k: int, threads: int, num_stages: int):
        block_m = 16

        @T.prim_func
        def main(
            a: T.Tensor(a_shape, dtype),
            weights: T.Tensor((num_experts, weight_n, k), dtype),
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            route_metadata: T.Tensor((metadata_size,), "int32"),
            out: T.Tensor((num_tokens, top_k, n), dtype),
        ):
            n_blocks = T.ceildiv(n, block_n)
            with T.Kernel(num_routes * n_blocks, threads=threads) as pid:
                route = pid // n_blocks
                n_block = pid % n_blocks
                route_count = T.alloc_var("int32", init=1)
                is_leader = T.alloc_var("int32", init=1)
                token = route // top_k
                slot = route % top_k
                expert = expert_ids[token, slot]
                if dispatch_mode == "grouped":
                    count = route_metadata[expert]
                    if count <= group_capacity:
                        route_count = count
                        if route_metadata[rank_base + route] != 0:
                            is_leader = 0

                a_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((2 * block_n if gated else block_n, block_k), dtype)
                acc = T.alloc_fragment((block_m, 2 * block_n if gated else block_n), "float32")
                acc_out = T.alloc_shared((block_m, 2 * block_n if gated else block_n), "float32")

                if is_leader != 0:
                    T.clear(acc)
                    for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                        if dispatch_mode == "direct" or route_count == 1:
                            for ki in T.Parallel(block_k):
                                offset = ko * block_k + ki
                                if offset < k:
                                    if route_input:
                                        a_shared[0, ki] = a[token, slot, offset]
                                    else:
                                        a_shared[0, ki] = a[token, offset]
                                else:
                                    a_shared[0, ki] = T.Cast(dtype, 0)
                        else:
                            for mi, ki in T.Parallel(block_m, block_k):
                                offset = ko * block_k + ki
                                if mi < route_count:
                                    selected_route = route_metadata[
                                        list_base + expert * group_capacity + mi
                                    ]
                                    selected_token = selected_route // top_k
                                    selected_slot = selected_route % top_k
                                    if offset < k:
                                        if route_input:
                                            a_shared[mi, ki] = a[
                                                selected_token, selected_slot, offset
                                            ]
                                        else:
                                            a_shared[mi, ki] = a[selected_token, offset]
                                    else:
                                        a_shared[mi, ki] = T.Cast(dtype, 0)
                        T.copy(
                            weights[
                                expert,
                                n_block * block_n : (n_block + 1) * block_n,
                                ko * block_k : (ko + 1) * block_k,
                            ],
                            weight_shared[0:block_n, :],
                            eviction_policy="evict_first",
                        )
                        if gated:
                            T.copy(
                                weights[
                                    expert,
                                    n + n_block * block_n : n + (n_block + 1) * block_n,
                                    ko * block_k : (ko + 1) * block_k,
                                ],
                                weight_shared[block_n : 2 * block_n, :],
                                eviction_policy="evict_first",
                            )
                        T.sync_threads()
                        T.gemm(a_shared, weight_shared, acc, transpose_B=True)

                    T.copy(acc, acc_out)
                    if dispatch_mode == "direct" or route_count == 1:
                        for j in T.Parallel(block_n):
                            col = n_block * block_n + j
                            if col < n:
                                if gated:
                                    gate = acc_out[0, j]
                                    value = gate * T.sigmoid(gate) * acc_out[0, block_n + j]
                                else:
                                    value = acc_out[0, j]
                                out[token, slot, col] = T.Cast(dtype, value)
                    else:
                        for mi, j in T.Parallel(block_m, block_n):
                            col = n_block * block_n + j
                            if mi < route_count and col < n:
                                selected_route = route_metadata[
                                    list_base + expert * group_capacity + mi
                                ]
                                selected_token = selected_route // top_k
                                selected_slot = selected_route % top_k
                                if gated:
                                    gate = acc_out[mi, j]
                                    value = gate * T.sigmoid(gate) * acc_out[mi, block_n + j]
                                else:
                                    value = acc_out[mi, j]
                                out[selected_token, selected_slot, col] = T.Cast(dtype, value)

        return main

    return _kernel


@functools.lru_cache(maxsize=32)
def _weighted_reduce(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    dtype: str,
    scaling: float,
):
    @tilelang.jit(out_idx=[])
    def _kernel(threads: int):
        @T.prim_func
        def main(
            expert_output: T.Tensor((num_tokens, top_k, hidden_size), dtype),
            topk_weights: T.Tensor((num_tokens, top_k), "float32"),
            output: T.Tensor((num_tokens, hidden_size), dtype),
        ):
            with T.Kernel(T.ceildiv(num_tokens * hidden_size, threads), threads=threads) as bid:
                tid = T.get_thread_binding()
                index = bid * threads + tid
                if index < num_tokens * hidden_size:
                    token = index // hidden_size
                    col = index % hidden_size
                    acc = T.alloc_local((1,), "float32")
                    acc[0] = 0
                    for slot in T.serial(top_k):
                        acc[0] += (
                            T.Cast("float32", expert_output[token, slot, col])
                            * topk_weights[token, slot]
                        )
                    output[token, col] = T.Cast(dtype, acc[0] * scaling)

        return main

    return _kernel


class IndexedRouteStatsKernel(Kernel):
    """Describe same-expert route groups on device."""

    supported_archs = [90]

    def __init__(self, num_tokens: int, top_k: int, num_experts: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.group_capacity = _group_capacity(num_tokens)
        self.kernel = _route_stats(num_tokens, top_k, num_experts, self.group_capacity)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def output_size(self) -> int:
        return self.required_output_size(self.num_tokens, self.top_k, self.num_experts)

    @staticmethod
    def required_output_size(num_tokens: int, top_k: int, num_experts: int) -> int:
        return _route_metadata_size(num_experts, num_tokens * top_k, _group_capacity(num_tokens))

    def forward(self, expert_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        expected = (self.num_tokens, self.top_k)
        if tuple(expert_ids.shape) != expected or expert_ids.dtype is not torch.int32:
            raise ValueError(f"expert_ids must be {expected} int32")
        if tuple(out.shape) != (self.output_size,) or out.dtype is not torch.int32:
            raise ValueError(f"out must be ({self.output_size},) int32")
        self.kernel(**self.config)(expert_ids, out)
        return out


class IndexedExpertGemmTemplate(Kernel):
    """Compute expert-selected rows directly from ``expert_ids``.

    ``a`` is ``[T,K]`` or ``[T,top_k,K]``; weights are ``[E,N,K]`` or
    ``[E,2N,K]`` for a gated activation; output is ``[T,top_k,N]``.
    """

    supported_archs = [90]

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        *,
        activation: str = "none",
        route_input: bool = False,
        dispatch_mode: str = "direct",
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if activation not in ("none", "silu_and_mul"):
            raise ValueError("activation must be 'none' or 'silu_and_mul'")
        if dispatch_mode not in ("direct", "grouped"):
            raise ValueError("dispatch_mode must be 'direct' or 'grouped'")
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.n = n
        self.k = k
        self.dtype = dtype
        self.activation = activation
        self.route_input = route_input
        self.dispatch_mode = dispatch_mode
        self.group_capacity = _group_capacity(num_tokens)
        self.kernel = _indexed_expert_gemm(
            num_tokens,
            top_k,
            num_experts,
            n,
            k,
            self.dtype_str,
            activation,
            route_input,
            dispatch_mode,
            self.group_capacity,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        block_n = (
            128 if self.route_input and (self.num_tokens == 1 or self.num_tokens >= 16) else 64
        )
        block_k = 128
        num_stages = 3 if 1 < self.num_tokens < 16 else 2
        if self.route_input and (
            self.num_tokens == 1 or (self.dispatch_mode == "grouped" and 4 <= self.num_tokens <= 8)
        ):
            block_k = 256
            num_stages = 2
        return {
            "block_n": block_n,
            "block_k": block_k,
            "threads": 128,
            "num_stages": num_stages,
        }

    def forward(
        self,
        a: torch.Tensor,
        weights: torch.Tensor,
        expert_ids: torch.Tensor,
        route_metadata: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected_a = (
            (self.num_tokens, self.top_k, self.k) if self.route_input else (self.num_tokens, self.k)
        )
        expected_weight_n = 2 * self.n if self.activation != "none" else self.n
        if tuple(a.shape) != expected_a:
            raise ValueError(f"a must have shape {expected_a}")
        if tuple(weights.shape) != (self.num_experts, expected_weight_n, self.k):
            raise ValueError(
                f"weights must have shape {(self.num_experts, expected_weight_n, self.k)}"
            )
        if expert_ids.shape != (self.num_tokens, self.top_k) or expert_ids.dtype is not torch.int32:
            raise ValueError("expert_ids must be [num_tokens, top_k] int32")
        if a.dtype is not self.dtype or weights.dtype is not self.dtype:
            raise TypeError("a and weights must match the template dtype")
        if route_metadata is None:
            if self.dispatch_mode == "grouped":
                raise ValueError("route_metadata is required for grouped dispatch")
            route_metadata = expert_ids.reshape(-1)[:1]
        expected_metadata = (
            _route_metadata_size(
                self.num_experts, self.num_tokens * self.top_k, self.group_capacity
            )
            if self.dispatch_mode == "grouped"
            else 1
        )
        if (
            tuple(route_metadata.shape) != (expected_metadata,)
            or route_metadata.dtype is not torch.int32
        ):
            raise ValueError(f"route_metadata must be ({expected_metadata},) int32")
        expected_out = (self.num_tokens, self.top_k, self.n)
        if out is None:
            out = a.new_empty(expected_out)
        if tuple(out.shape) != expected_out or out.dtype is not self.dtype:
            raise ValueError(f"out must be {expected_out} with dtype {self.dtype}")
        self.kernel(**self.config)(a, weights, expert_ids, route_metadata, out)
        return out


class IndexedWeightedReduceKernel(Kernel):
    """Apply routing weights and reduce route-major expert output to tokens."""

    supported_archs = [90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        dtype: torch.dtype,
        scaling: float,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.kernel = _weighted_reduce(
            num_tokens, top_k, hidden_size, self.dtype_str, float(scaling)
        )
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"threads": 256}

    def forward(
        self,
        expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(expert_output.shape) != (self.num_tokens, self.top_k, self.hidden_size):
            raise ValueError("expert_output has the wrong shape")
        if tuple(topk_weights.shape) != (self.num_tokens, self.top_k):
            raise ValueError("topk_weights has the wrong shape")
        if tuple(output.shape) != (self.num_tokens, self.hidden_size):
            raise ValueError("output has the wrong shape")
        if expert_output.dtype is not self.dtype or output.dtype is not self.dtype:
            raise TypeError("expert_output and output must match the kernel dtype")
        if topk_weights.dtype is not torch.float32:
            raise TypeError("topk_weights must be float32")
        self.kernel(**self.config)(expert_output, topk_weights, output)
        return output
