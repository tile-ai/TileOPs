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

correction_bias variant (with_correction_bias=True):
  Adds a per-expert bias to sigmoid scores before top-k selection, while
  writing the original (unbiased) sigmoid score to topk_weights.
  Used by Kimi K2 / DeepSeekV3-variant models.

  Extra register array my_biased[ELEMS_PER_THREAD] = sigmoid(logit) + bias.
  K-pass argmax compares my_biased; output uses my_scores (unbiased sigmoid).
  Warp all-reduce tracks both l_best_val (biased, for tie-breaking) and
  l_best_orig (unbiased, for output) via paired shfl_xor.

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
  "sigmoid" — element-wise sigmoid (DeepSeek-V3, GLM-4, Kimi K2)

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


@functools.lru_cache(maxsize=64)
def _fused_topk_kernel(num_tokens, num_experts, top_k, scoring_func, with_correction_bias=False):
    """Build a fused TileLang kernel: scoring + top-k, zero __syncthreads().

    Args:
        num_tokens: T — number of tokens.
        num_experts: E — number of experts.
        top_k: K — experts to select per token.
        scoring_func: "softmax" or "sigmoid" (compile-time constant).
        with_correction_bias: If True, accept a per-expert bias tensor and add it
            to sigmoid scores for top-k selection (bias does NOT affect output
            weights, which remain the original sigmoid scores).

    Returns:
        JIT factory _func(TOKENS_PER_BLOCK) → callable.
    """

    @tilelang.jit(out_idx=[])
    def _func(TOKENS_PER_BLOCK):
        WARP_SIZE = _WARP_SIZE
        # Each lane handles ceil(E / 32) experts stored in local registers.
        ELEMS_PER_THREAD = -(-num_experts // WARP_SIZE)  # ceildiv(E, 32)
        LOG_WARP = int(math.log2(WARP_SIZE))    # = 5 for WARP_SIZE=32
        HALF_WARP = WARP_SIZE // 2              # = 16
        num_blocks = -(-num_tokens // TOKENS_PER_BLOCK)  # ceildiv(T, TPB)

        if with_correction_bias:
            # ── Variant: sigmoid + per-expert correction bias ─────────────────
            # Kernel signature adds correction_bias [E] float32.
            # K-pass argmax selects based on (sigmoid + bias); output writes
            # the original (unbiased) sigmoid score.
            @T.prim_func
            def main(
                gating_output: T.Tensor([num_tokens, num_experts], "float32"),   # noqa: F821
                correction_bias: T.Tensor([num_experts], "float32"),             # noqa: F821
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),          # noqa: F821
                topk_ids: T.Tensor([num_tokens, top_k], "int32"),                # noqa: F821
            ):
                with T.Kernel(num_blocks, threads=TOKENS_PER_BLOCK * WARP_SIZE) as (block_id,):
                    tx = T.get_thread_binding()
                    warp_id = tx // WARP_SIZE
                    lane_id = tx % WARP_SIZE
                    token_id = block_id * TOKENS_PER_BLOCK + warp_id

                    # my_scores[j]:  original sigmoid(logit)  — for output weight
                    # my_biased[j]:  sigmoid(logit) + bias    — for argmax selection
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")
                    my_biased = T.alloc_local([ELEMS_PER_THREAD], "float32")

                    if token_id < num_tokens:

                        # ── Step 1: Load raw logits ───────────────────────────
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        # ── Step 2: Sigmoid + bias (element-wise, no syncs) ───
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                val = my_scores[j]
                                sig_val = T.float32(1) / (T.float32(1) + T.exp(-val))
                                my_scores[j] = sig_val
                                my_biased[j] = sig_val + correction_bias[expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")
                                my_biased[j] = -T.infinity("float32")

                        # ── Step 3: K-pass argmax (zero syncs) ────────────────
                        #
                        # Compare using my_biased (biased) for expert selection.
                        # Write my_scores (original sigmoid) to topk_weights.
                        # Track l_best_orig alongside l_best_val through the
                        # warp shfl_xor all-reduce so lane 0 has the correct
                        # unbiased weight to write.
                        l_best_val = T.alloc_var(T.float32)   # biased (for selection)
                        l_best_orig = T.alloc_var(T.float32)  # original sigmoid (for output)
                        l_best_idx = T.alloc_var(T.int32)

                        for k in T.serial(top_k):
                            # Lane-local argmax on biased scores
                            l_best_val = -T.infinity("float32")
                            l_best_orig = T.float32(0)
                            l_best_idx = T.int32(-1)
                            for j in T.serial(ELEMS_PER_THREAD):
                                if my_biased[j] > l_best_val:
                                    l_best_val = my_biased[j]
                                    l_best_orig = my_scores[j]
                                    l_best_idx = j * T.int32(WARP_SIZE) + lane_id

                            # Warp (biased_val, orig_val, idx) all-reduce via shfl_xor.
                            # Winner decided by biased_val with tie-breaking on lower idx.
                            # l_best_orig is carried along so all lanes end up with the
                            # original sigmoid score of the winning expert.
                            for i in T.serial(LOG_WARP):
                                mask = T.int32(HALF_WARP) >> i
                                other_val = T.shfl_xor(l_best_val, mask)
                                other_orig = T.shfl_xor(l_best_orig, mask)
                                other_idx = T.shfl_xor(l_best_idx, mask)
                                # Update orig first (uses old l_best_val / l_best_idx)
                                l_best_orig = T.if_then_else(
                                    other_val > l_best_val,
                                    other_orig,
                                    T.if_then_else(
                                        other_val == l_best_val,
                                        T.if_then_else(
                                            other_idx < l_best_idx, other_orig, l_best_orig
                                        ),
                                        l_best_orig,
                                    ),
                                )
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

                            # Lane 0 writes original (unbiased) sigmoid score
                            if lane_id == 0:
                                topk_weights[token_id, k] = l_best_orig
                                topk_ids[token_id, k] = l_best_idx

                            # Mask both arrays so this expert is not selected again
                            for j in T.serial(ELEMS_PER_THREAD):
                                if j * T.int32(WARP_SIZE) + lane_id == l_best_idx:
                                    my_scores[j] = -T.infinity("float32")
                                    my_biased[j] = -T.infinity("float32")

        else:
            # ── Standard variant (no correction bias) ─────────────────────────
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

                    # ── Local register array ──────────────────────────────────
                    my_scores = T.alloc_local([ELEMS_PER_THREAD], "float32")

                    # ── Guard: skip warps whose token_id is out of range ──────
                    if token_id < num_tokens:

                        # ── Step 1: Load experts into registers ───────────────
                        for j in T.serial(ELEMS_PER_THREAD):
                            expert_idx = j * WARP_SIZE + lane_id
                            if expert_idx < num_experts:
                                my_scores[j] = gating_output[token_id, expert_idx]
                            else:
                                my_scores[j] = -T.infinity("float32")

                        # ── Step 2: Scoring (zero __syncthreads, warp shfl only)
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

                            # Exp in-place + warp sum all-reduce (shfl_xor, no barrier)
                            l_sum = T.float32(0)
                            for j in T.serial(ELEMS_PER_THREAD):
                                val = T.exp(my_scores[j] - l_max)
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

                        # ── Step 3: K-pass argmax (zero __syncthreads, warp shfl)
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
        with_correction_bias: If True, accept a per-expert correction_bias tensor in
            forward(). Adds bias to sigmoid scores for expert selection while writing
            unbiased sigmoid scores to topk_weights. Requires scoring_func="sigmoid".
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
        with_correction_bias: bool = False,
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
        if with_correction_bias and scoring_func != "sigmoid":
            raise ValueError(
                "with_correction_bias=True requires scoring_func='sigmoid'. "
                f"Got scoring_func='{scoring_func}'."
            )

        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.with_correction_bias = with_correction_bias

        self._kernel_fn = _fused_topk_kernel(
            num_tokens, num_experts, top_k, scoring_func, with_correction_bias
        )
        self.init_config(config, tune=False)

    @property
    def default_config(self) -> dict:
        # TOKENS_PER_BLOCK = 16 gives 512 threads/block (16 warps/block).
        return {"TOKENS_PER_BLOCK": 16}

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run scoring + top-k selection.

        Args:
            gating_output: [T, E] router logits, any float dtype.
            correction_bias: [E] float32 per-expert bias (required when
                with_correction_bias=True, must be None otherwise).

        Returns:
            topk_weights: [T, K] float32 routing weights.
            topk_ids:     [T, K] int32 expert indices.
        """
        assert gating_output.shape == (self.num_tokens, self.num_experts), (
            f"Expected gating_output shape ({self.num_tokens}, {self.num_experts}), "
            f"got {tuple(gating_output.shape)}"
        )
        assert gating_output.is_cuda, "gating_output must be on CUDA"

        if self.with_correction_bias:
            assert correction_bias is not None, (
                "correction_bias must be provided when with_correction_bias=True"
            )
            assert correction_bias.shape == (self.num_experts,), (
                f"Expected correction_bias shape ({self.num_experts},), "
                f"got {tuple(correction_bias.shape)}"
            )
        else:
            assert correction_bias is None, (
                "correction_bias must be None when with_correction_bias=False"
            )

        logits_f32 = gating_output.to(torch.float32)

        dev = logits_f32.device
        topk_weights = torch.empty(self.num_tokens, self.top_k, dtype=torch.float32, device=dev)
        topk_ids = torch.empty(self.num_tokens, self.top_k, dtype=torch.int32, device=dev)

        fn = self._kernel_fn(self.config["TOKENS_PER_BLOCK"])
        if self.with_correction_bias:
            bias_f32 = correction_bias.to(torch.float32)
            fn(logits_f32, bias_f32, topk_weights, topk_ids)
        else:
            fn(logits_f32, topk_weights, topk_ids)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_ids
