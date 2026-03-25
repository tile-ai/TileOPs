"""MoE fused top-k routing kernel (fused scoring + top-k selection).

Single TileLang kernel fusing scoring and top-k into one pass (no global memory
roundtrip for intermediate scores).

Algorithm (TOKENS_PER_BLOCK tokens per block, 1 warp per token):
  Each warp of 32 lanes independently handles one token.
  Thread lane l holds experts {l, l+32, l+64, ...} in local registers.

  1. Load: each lane reads ceil(E/32) experts from global memory into registers.
     Padding elements (idx >= E) are initialized to -inf.
  2. Scoring:
       softmax → warp-level 2-pass (max shfl-reduce, exp+sum shfl-reduce); no syncs
       sigmoid → element-wise sigmoid in-register; no syncs
  3. K-pass argmax: for each of K iterations:
       - Lane-local argmax over ceil(E/32) register elements
       - Warp-level (val, idx) all-reduce via shfl_xor (no barrier)
       - Lane 0 writes topk_weights[token_id, k] and topk_ids[token_id, k]
       - All lanes independently mask their selected expert to -inf in registers

Barrier analysis:
  All reduces are intra-warp (shfl_xor, no __syncthreads).
  ZERO __syncthreads() calls for all paths.
  vs old 1-block-per-token: 22 syncs (softmax), 18 syncs (sigmoid)
  vs vLLM (CUB BlockReduce, 2 kernels): ~29 syncs

Grid/occupancy (T=4096, E=256, TOKENS_PER_BLOCK=16):
  Grid = ceil(T / TOKENS_PER_BLOCK) = 256 blocks
  Threads per block = TOKENS_PER_BLOCK * 32 = 512
  Max blocks/SM = 2048 / 512 = 4  →  4 * 132 = 528 concurrent blocks
  256 < 528 → all blocks run in 1 wave

TOKENS_PER_BLOCK is the tunable parameter (default 16; try 4 or 8 for small T).

Supported scoring functions:
  "softmax" — row-wise softmax (Qwen3, Qwen2, Qwen3.5)
  "sigmoid" — element-wise sigmoid (DeepSeek-V3, GLM-4)

Outputs:
  topk_weights  [T, K]  float32 — routing weights (optionally renormalized)
  topk_ids      [T, K]  int32   — expert indices
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["FusedTopKKernel"]

_SCORING_FUNCS = ("softmax", "sigmoid")
_WARP_SIZE = 32


@functools.lru_cache(maxsize=32)
def _fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func):
    """Build a fused TileLang kernel: scoring + top-k, zero __syncthreads().

    Args:
        num_tokens: T — number of tokens.
        num_experts: E — number of experts.
        top_k: K — experts to select per token.
        scoring_func: "softmax" or "sigmoid" (compile-time constant).

    Returns:
        JIT factory _func(TOKENS_PER_BLOCK) → callable(gating_output, topk_weights, topk_ids).
    """

    @tilelang.jit(out_idx=[])
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = _WARP_SIZE
        # Each lane handles ceil(E / 32) experts stored in local registers.
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)  # ceildiv(E, 32)
        LOG_WARP = int(math.log2(WARP_SIZE))    # = 5 for WARP_SIZE=32
        HALF_WARP = WARP_SIZE // 2              # = 16
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)  # ceildiv(T, TPB)

        @T.prim_func
        def main(
            gating_output: T.Tensor([num_tokens, num_experts], "float32"),  # noqa: F821
            topk_weights: T.Tensor([num_tokens, top_k], "float32"),         # noqa: F821
            topk_ids: T.Tensor([num_tokens, top_k], "int32"),               # noqa: F821
        ):
            with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                tx = T.get_thread_binding()
                warp_id = tx // WARP_SIZE
                lane_id = tx % WARP_SIZE
                # Each warp handles one token.
                token_id = block_id * TOKENS_PER_BLOCK + warp_id

                # ── Local register array ─────────────────────────────────────────
                # Lane l owns experts: {l + j*32 | j=0..ELEMS_PER_THREAD-1}.
                # No shared memory — all communication is intra-warp via shfl_xor.
                my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")

                # ── Guard: skip warps whose token_id is out of range ────────────
                # (happens when num_tokens % TOKENS_PER_BLOCK != 0)
                if token_id < num_tokens:

                    # ── Step 1: Load experts into registers ──────────────────────
                    # Padding (expert_idx >= num_experts) → -inf so argmax skips.
                    for j in T.serial(ELEMS_PER_THREAD):
                        expert_idx = j * WARP_SIZE + lane_id
                        if expert_idx < num_experts:
                            my_scores[j] = gating_output[token_id, expert_idx]
                        else:
                            my_scores[j] = -T.infinity("float32")

                    # ── Step 2: Scoring (zero __syncthreads, warp shfl only) ─────
                    inv_row_sum = T.alloc_var(T.float32)
                    inv_row_sum = T.float32(1)  # default (sigmoid / no renorm)

                    if scoring_func == "softmax":
                        l_max = T.alloc_var(T.float32)
                        l_sum = T.alloc_var(T.float32)

                        # Warp max all-reduce (shfl_xor, no barrier)
                        l_max = -T.infinity("float32")
                        for j in T.serial(ELEMS_PER_THREAD):
                            l_max = T.max(l_max, my_scores[j])
                        for i in T.serial(LOG_WARP):
                            l_max = T.max(l_max, T.shfl_xor(l_max, T.int32(HALF_WARP) >> i))
                        # All lanes now hold the same l_max (= row_max) via shfl_xor broadcast.

                        # Exp in-place + warp sum all-reduce (shfl_xor, no barrier)
                        l_sum = T.float32(0)
                        for j in T.serial(ELEMS_PER_THREAD):
                            val = T.exp(my_scores[j] - l_max)
                            # exp(-inf - max) = 0; padding contributes 0 to sum.
                            my_scores[j] = val
                            l_sum = l_sum + val
                        for i in T.serial(LOG_WARP):
                            l_sum = l_sum + T.shfl_xor(l_sum, T.int32(HALF_WARP) >> i)
                        inv_row_sum = T.float32(1) / l_sum

                    else:  # sigmoid: element-wise, no row reduction
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                val = my_scores[j]
                                my_scores[j] = T.float32(1) / (T.float32(1) + T.exp(-val))
                            # Padding already -inf; sigmoid(-inf)≈0, keep -inf for argmax.

                    # ── Step 3: K-pass argmax (zero __syncthreads, warp shfl) ─────
                    #
                    # Per k-iteration:
                    #   1. Lane-local argmax over ELEMS_PER_THREAD registers
                    #   2. Warp shfl_xor all-reduce → every lane has (best_val, best_idx)
                    #   3. Lane 0 writes output
                    #   4. All lanes independently mask their register (no sync needed)
                    #
                    # After shfl_xor all-reduce, every lane holds the same best_idx.
                    # The lane owning best_idx (lane = best_idx % 32) sets its register
                    # entry j = best_idx // 32 to -inf.  All other lanes' checks fail.
                    l_best_val = T.alloc_var(T.float32)
                    l_best_idx = T.alloc_var(T.int32)

                    for k in T.serial(top_k):
                        # Lane-local argmax
                        l_best_val = -T.infinity("float32")
                        l_best_idx = T.int32(-1)
                        for j in T.serial(ELEMS_PER_THREAD):
                            if my_scores[j] > l_best_val:
                                l_best_val = my_scores[j]
                                l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                        # Warp (val, idx) all-reduce via shfl_xor (no barrier).
                        # Implements lexicographic comparison on (-val, idx): the lane
                        # with the higher value wins; ties are broken by the lower expert
                        # index. This guarantees all 32 lanes converge to the same
                        # (best_val, best_idx) after 5 rounds, even when bf16 inputs
                        # produce identical float32 scores for two experts — preventing
                        # split-brain masking where two lanes mask different experts.
                        #
                        # Nested T.if_then_else encodes:
                        #   if other_val > l_best_val:   take other   (higher value wins)
                        #   elif other_val == l_best_val
                        #        and other_idx < l_best_idx: take other (lower index wins)
                        #   else:                         keep current
                        for i in T.serial(LOG_WARP):
                            mask = T.int32(HALF_WARP) >> i
                            other_val = T.shfl_xor(l_best_val, mask)
                            other_idx = T.shfl_xor(l_best_idx, mask)
                            l_best_idx = T.if_then_else(
                                other_val > l_best_val,
                                other_idx,
                                T.if_then_else(
                                    other_val == l_best_val,
                                    T.if_then_else(
                                        other_idx < l_best_idx, other_idx, l_best_idx
                                    ),
                                    l_best_idx,
                                ),
                            )
                            l_best_val = T.max(l_best_val, other_val)

                        # Lane 0 writes output
                        if lane_id == 0:
                            topk_weights[token_id, k] = l_best_val * inv_row_sum
                            topk_ids[token_id, k] = l_best_idx

                        # All lanes mask their own register entry — NO sync needed.
                        # After shfl_xor, every lane has the same l_best_idx.
                        # Exactly one lane (lane = l_best_idx % 32) will match
                        # for exactly one j (j = l_best_idx // 32).
                        for j in T.serial(ELEMS_PER_THREAD):
                            if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                my_scores[j] = -T.infinity("float32")

        return main

    return _func


class FusedTopKKernel(Kernel):
    """MoE top-k routing kernel — fused scoring + top-k, zero __syncthreads().

    Uses a per-warp algorithm: each warp of 32 lanes independently handles one
    token, keeping expert scores in local registers.  All reductions (softmax
    max/sum, argmax) use warp shfl_xor — no shared memory, no __syncthreads__.

    Barrier count: 0 (vs 22 syncs for the old 1-block-per-token design).

    Args:
        num_tokens: Number of input tokens T.
        num_experts: Number of experts E.
        top_k: Number of experts to select per token K.
        scoring_func: "softmax" or "sigmoid".
        renormalize: If True, normalize selected weights to sum to 1 (done in PyTorch).
        config: Optional kernel config dict (key: "TOKENS_PER_BLOCK").
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        renormalize: bool = False,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if scoring_func not in _SCORING_FUNCS:
            raise ValueError(
                f"Unsupported scoring_func '{scoring_func}'. "
                f"Expected one of {_SCORING_FUNCS}."
            )
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) must be <= num_experts ({num_experts})")

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize

        self._kernel_fn = _fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func)
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        # TOKENS_PER_BLOCK = 16 gives 512 threads/block (16 warps/block).
        # For T=4096: 256 blocks → 2 blocks/SM → all fit in 1-2 waves.
        # Empirically faster than 4 due to reduced block-scheduling overhead.
        return {"TOKENS_PER_BLOCK": 16}

    def forward(
        self,
        gating_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scoring + top-k selection.

        Args:
            gating_output: [T, E] router logits, any float dtype.

        Returns:
            topk_weights: [T, K] float32 routing weights.
            topk_ids:     [T, K] int32 expert indices.
        """
        assert gating_output.shape == (self.num_tokens, self.num_experts), (
            f"Expected gating_output shape ({self.num_tokens}, {self.num_experts}), "
            f"got {tuple(gating_output.shape)}"
        )
        assert gating_output.is_cuda, "gating_output must be on CUDA"

        logits_f32 = gating_output.to(torch.float32)

        dev = logits_f32.device
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)

        fn = self._kernel_fn(self.config["TOKENS_PER_BLOCK"])
        fn(logits_f32, topk_weights, topk_ids)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids
