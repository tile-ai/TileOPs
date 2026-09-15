"""Route-indexed small-M expert MLP kernels."""

import functools
from typing import Optional

import tilelang
import tilelang.language as T
import torch
from tilelang.transform import PassConfigKey

from tileops.kernels.kernel_base import Kernel

__all__ = ["IndexedExpertGemmTemplate"]


def _route_metadata_size(num_routes: int) -> int:
    return 2 + 2 * num_routes + 16 * num_routes


@functools.lru_cache(maxsize=32)
def _route_stats(num_tokens: int, top_k: int):
    num_routes = num_tokens * top_k
    list_base = 2 + 2 * num_routes

    @tilelang.jit(out_idx=[])
    def _kernel(threads: int):
        @T.prim_func
        def main(
            expert_ids: T.Tensor((num_tokens, top_k), "int32"),
            metadata: T.Tensor((_route_metadata_size(num_routes),), "int32"),
        ):
            with T.Kernel(1, threads=threads):
                tid = T.get_thread_binding()
                for route in T.Parallel(num_routes):
                    expert = expert_ids[route // top_k, route % top_k]
                    count = T.alloc_local((1,), "int32")
                    rank = T.alloc_local((1,), "int32")
                    count[0] = 0
                    rank[0] = 0
                    for candidate in T.serial(num_routes):
                        if expert_ids[candidate // top_k, candidate % top_k] == expert:
                            if count[0] < 16:
                                metadata[list_base + route * 16 + count[0]] = candidate
                            count[0] += 1
                            if candidate < route:
                                rank[0] += 1
                    metadata[2 + route] = count[0]
                    metadata[2 + num_routes + route] = rank[0]
                T.sync_threads()
                if tid == 0:
                    active = T.alloc_local((1,), "int32")
                    max_count = T.alloc_local((1,), "int32")
                    active[0] = 0
                    max_count[0] = 0
                    for route in T.serial(num_routes):
                        if metadata[2 + num_routes + route] == 0:
                            active[0] += 1
                        max_count[0] = T.max(max_count[0], metadata[2 + route])
                    metadata[0] = active[0]
                    metadata[1] = max_count[0]

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
    dispatch_by_reuse: bool,
    dispatch_mode: str,
    reuse_numerator: int,
    reuse_denominator: int,
):
    num_routes = num_tokens * top_k
    gated = activation != "none"
    weight_n = 2 * n if gated else n
    a_shape = (num_tokens, top_k, k) if route_input else (num_tokens, k)
    metadata_size = _route_metadata_size(num_routes) if dispatch_by_reuse else 1
    list_base = 2 + 2 * num_routes

    @tilelang.jit(
        out_idx=[],
        compile_flags=["-O3", "-DENABLE_BF16"],
        # The A tile is rewritten inside the K loop, so producer/consumer WS would race.
        pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    def _kernel(block_m: int, block_n: int, block_k: int, threads: int, num_stages: int):
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
                token = route // top_k
                slot = route % top_k
                expert = expert_ids[token, slot]
                route_count = T.alloc_local((1,), "int32")
                is_leader = T.alloc_local((1,), "int32")
                route_count[0] = 1
                is_leader[0] = 1
                if dispatch_by_reuse:
                    should_group = (
                        num_routes * reuse_denominator >= route_metadata[0] * reuse_numerator
                        and route_metadata[1] <= 16
                    )
                    if dispatch_mode == "direct" and should_group:
                        is_leader[0] = 0
                    if dispatch_mode == "grouped":
                        if should_group:
                            route_count[0] = route_metadata[2 + route]
                            if route_metadata[2 + num_routes + route] != 0:
                                is_leader[0] = 0
                        else:
                            is_leader[0] = 0

                a_shared = T.alloc_shared((block_m, block_k), dtype)
                gate_shared = T.alloc_shared((block_n, block_k), dtype)
                gate_acc = T.alloc_fragment((block_m, block_n), "float32")
                gate_out = T.alloc_shared((block_m, block_n), "float32")
                if dispatch_mode == "grouped":
                    route_indices = T.alloc_local((16,), "int32")
                if gated:
                    up_shared = T.alloc_shared((block_n, block_k), dtype)
                    up_acc = T.alloc_fragment((block_m, block_n), "float32")
                    up_out = T.alloc_shared((block_m, block_n), "float32")

                if is_leader[0] != 0:
                    if dispatch_mode == "grouped":
                        for mi in T.serial(16):
                            if mi < route_count[0]:
                                route_indices[mi] = route_metadata[list_base + route * 16 + mi]
                    T.clear(gate_acc)
                    if gated:
                        T.clear(up_acc)
                    for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                        for mi, ki in T.Parallel(block_m, block_k):
                            offset = ko * block_k + ki
                            if mi < (route_count[0] if dispatch_mode == "grouped" else 1):
                                selected_route = T.alloc_var("int32", init=route)
                                if dispatch_mode == "grouped":
                                    selected_route = route_indices[mi]
                                selected_token = selected_route // top_k
                                selected_slot = selected_route % top_k
                                if offset < k:
                                    if route_input:
                                        a_shared[mi, ki] = a[selected_token, selected_slot, offset]
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
                            gate_shared,
                        )
                        if gated:
                            T.copy(
                                weights[
                                    expert,
                                    n + n_block * block_n : n + (n_block + 1) * block_n,
                                    ko * block_k : (ko + 1) * block_k,
                                ],
                                up_shared,
                            )
                        T.sync_threads()
                        T.gemm(a_shared, gate_shared, gate_acc, transpose_B=True)
                        if gated:
                            T.gemm(a_shared, up_shared, up_acc, transpose_B=True)

                    T.copy(gate_acc, gate_out)
                    if gated:
                        T.copy(up_acc, up_out)
                    for mi, j in T.Parallel(block_m, block_n):
                        col = n_block * block_n + j
                        if mi < (route_count[0] if dispatch_mode == "grouped" else 1) and col < n:
                            selected_route = T.alloc_var("int32", init=route)
                            if dispatch_mode == "grouped":
                                selected_route = route_indices[mi]
                            selected_token = selected_route // top_k
                            selected_slot = selected_route % top_k
                            if gated:
                                gate = gate_out[mi, j]
                                value = gate * T.sigmoid(gate) * up_out[mi, j]
                            else:
                                value = gate_out[mi, j]
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
    """Describe active experts and same-expert route groups on device."""

    supported_archs = [90]

    def __init__(self, num_tokens: int, top_k: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.kernel = _route_stats(num_tokens, top_k)
        self.init_config()

    @property
    def default_config(self) -> dict:
        return {"threads": 128}

    @property
    def output_size(self) -> int:
        return self.required_output_size(self.num_tokens, self.top_k)

    @staticmethod
    def required_output_size(num_tokens: int, top_k: int) -> int:
        return _route_metadata_size(num_tokens * top_k)

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
        dispatch_by_reuse: bool = False,
        dispatch_mode: str = "direct",
        reuse_threshold: tuple[int, int] = (8, 7),
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if activation not in ("none", "silu_and_mul"):
            raise ValueError("activation must be 'none' or 'silu_and_mul'")
        if dispatch_mode not in ("direct", "grouped"):
            raise ValueError("dispatch_mode must be 'direct' or 'grouped'")
        if dispatch_mode == "grouped" and not dispatch_by_reuse:
            raise ValueError("grouped dispatch_mode requires dispatch_by_reuse")
        numerator, denominator = reuse_threshold
        if numerator <= denominator or denominator <= 0:
            raise ValueError("reuse_threshold must be a ratio greater than one")
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.n = n
        self.k = k
        self.dtype = dtype
        self.activation = activation
        self.route_input = route_input
        self.dispatch_by_reuse = dispatch_by_reuse
        self.dispatch_mode = dispatch_mode
        self.reuse_threshold = reuse_threshold
        self.kernel = _indexed_expert_gemm(
            num_tokens,
            top_k,
            num_experts,
            n,
            k,
            self.dtype_str,
            activation,
            route_input,
            dispatch_by_reuse,
            dispatch_mode,
            numerator,
            denominator,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        block_m = 16
        if self.route_input and self.n % 128 == 0 and self.k % 256 == 0:
            return {
                "block_m": block_m,
                "block_n": 128,
                "block_k": 128 if self.dispatch_mode == "grouped" else 256,
                "threads": 128,
                "num_stages": 2,
            }
        return {
            "block_m": block_m,
            "block_n": 64,
            "block_k": 128,
            "threads": 128,
            "num_stages": 2,
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
            if self.dispatch_by_reuse:
                raise ValueError("route_metadata is required when dispatch_by_reuse is enabled")
            route_metadata = expert_ids.reshape(-1)[:1]
        expected_metadata = (
            _route_metadata_size(self.num_tokens * self.top_k) if self.dispatch_by_reuse else 1
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
