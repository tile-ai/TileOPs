"""MoE token permute kernel (no-pad variant).

Counting sort + tight gather: routes tokens to expert-contiguous layout
WITHOUT block_m-aligned padding. Each expert's tokens are stored contiguously
in the output buffer [T*K, H] with no inter-expert gaps.

Phase 1 (1 block, 1024 threads):
  - Count tokens per expert
  - Exclusive prefix-sum → expert_first_token_offset [E+1] (int64)
  - Compute tight layout: true_offsets [E] (int32), true_sizes [E] (int32)
  - Build permuted_idx [T*K] and fwd_idx [T*K]:
      permuted_idx[slot] = token row (for gather)
      fwd_idx[flat_idx] = tight slot (for unpermute)

Phase 2 (N blocks, 16 threads):
  - Gather hidden_states rows into perm_h at tight positions
  - Each thread loads 8 bf16 (128-bit) via uint4 vectorized ld.global

Outputs:
  perm_h                    [T*K, H]    tokens in tight expert-contiguous order
  true_offsets              [E]         int32 tight start per expert
  true_sizes                [E]         int32 true token count per expert
  expert_first_token_offset [E+1]       int64 non-padded exclusive prefix-sum
  fwd_idx                   [T*K]       int32 forward mapping: flat pos → tight slot
"""

from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MoePermuteNopadKernel"]

_SCAN_THREADS = 1024


def _make_scan_kernel_nopad(numel: int, num_experts: int, top_k: int):
    """Phase 1: count → prefix-sum → tight layout → fwd_idx (no block_m padding)."""

    @tilelang.jit(out_idx=[], compile_flags=["-O3"])
    def _scan(threads: int):
        @T.prim_func
        def _scan_main(
            flat_ids: T.Tensor([numel], "int32"),
            expert_first_token_offset: T.Tensor([num_experts + 1], "int64"),
            true_offsets: T.Tensor([num_experts], "int32"),
            true_sizes: T.Tensor([num_experts], "int32"),
            permuted_idx: T.Tensor([numel], "int32"),
            fwd_idx: T.Tensor([numel], "int32"),
            write_offsets: T.Tensor([num_experts], "int32"),
        ):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()

                s_counts = T.alloc_shared([num_experts], "int32")
                s_offset = T.alloc_shared([num_experts + 1], "int64")
                slot_buf = T.alloc_local([1], "int32")

                # Step 1: zero counts
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        s_counts[idx] = T.int32(0)
                T.sync_threads()

                # Step 2: count tokens per expert
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(s_counts[flat_ids[idx]], 1)
                T.sync_threads()

                # Step 3: exclusive prefix-sum → s_offset [E+1]
                if tx == 0:
                    s_offset[0] = T.int64(0)
                    for e in T.serial(num_experts):
                        s_offset[e + 1] = s_offset[e] + T.Cast(T.int64, s_counts[e])
                T.sync_threads()

                # Step 4: write outputs and init write_offsets (tight, no block_m alignment)
                for i in T.serial(T.ceildiv(num_experts + 1, threads)):
                    idx = i * threads + tx
                    if idx <= num_experts:
                        expert_first_token_offset[idx] = s_offset[idx]
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        true_offsets[idx] = T.Cast(T.int32, s_offset[idx])
                        true_sizes[idx] = s_counts[idx]
                        write_offsets[idx] = T.Cast(T.int32, s_offset[idx])
                T.sync_threads()

                # Step 5: scatter — assign each (token, topk) pair to its tight slot
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        eid = flat_ids[idx]
                        # Keep the returned slot in a local buffer: TileLang may otherwise
                        # duplicate the atomic while lowering bounds checks for the stores.
                        slot_buf[0] = T.atomic_add(write_offsets[eid], T.int32(1), return_prev=True)
                        slot = slot_buf[0]
                        permuted_idx[slot] = idx // T.int32(top_k)
                        # For no-pad: tight slot IS the fwd_idx (no padding offset)
                        fwd_idx[idx] = slot

        return _scan_main

    return _scan


def _make_gather_kernel_nopad(num_tokens: int, numel: int, hidden_size: int, dtype: str):
    """Phase 2: gather hidden_states rows into perm_h at tight positions.

    ``ROWS_PER_BLOCK`` slots per block (each thread handles 8 elements via a
    128-bit uint4 load/store). Grouping slots per block amortizes the launch /
    scheduling overhead of the one-block-per-slot layout, which dominated at
    prefill scale (numel up to ~32K blocks): on H200 this cut the gather from
    1.30x to 1.06x the HBM-bandwidth floor (~17% faster, matching torch's
    advanced-index gather). ROWS_PER_BLOCK=8 is the sweep optimum; >=32 degrades
    (too few blocks). threads = hidden_size // 8, capped at 1024.
    """
    VEC = 8
    threads = min(1024, hidden_size // VEC)
    while threads > 0 and hidden_size % threads != 0:
        threads -= 1
    threads = max(threads, 1)

    ROWS_PER_BLOCK = 8
    grid = (numel + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK

    @tilelang.jit(out_idx=[], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gather():
        @T.prim_func
        def _gather_main(
            hidden_states: T.Tensor([num_tokens, hidden_size], dtype),
            permuted_idx: T.Tensor([numel], "int32"),
            perm_h: T.Tensor([numel, hidden_size], dtype),
        ):
            with T.Kernel(grid, threads=threads) as (bid,):
                for r in T.serial(ROWS_PER_BLOCK):
                    slot = bid * ROWS_PER_BLOCK + r
                    if slot < numel:
                        # Direct global->global vectorized copy (no register-fragment
                        # round-trip; TileLang lowers both identically).
                        T.copy(
                            hidden_states[permuted_idx[slot], 0:hidden_size],
                            perm_h[slot, 0:hidden_size],
                        )

        return _gather_main

    return _gather


class MoePermuteNopadKernel(Kernel):
    """MoE token permute kernel without block_m-aligned padding.

    Sorts rank-grouped local assignments into tight expert-contiguous order.
    Output ``perm_h`` has exactly ``T*K`` rows. Global placement and
    communication are completed by EPDispatch before this kernel.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        num_experts: Number of local experts E.
        hidden_size: Hidden dimension H.
        dtype: Data type of hidden_states.
        config: Optional config dict with "threads".
        tune: Whether to autotune. The scan thread count is fixed by the
            single-block prefix-sum algorithm, so ``autotune_configs`` is
            undefined and ``tune=True`` degrades to the default config with a
            warning from ``Kernel.init_config``.

    Example:
        ```python linenums="1"
        kernel = MoePermuteNopadKernel(num_tokens=4, top_k=2, num_experts=8, hidden_size=128)
        perm_h, offsets, sizes, expert_offset, fwd_idx = kernel(hidden_states, topk_ids)
        ```
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.numel = num_tokens * top_k
        self._scan_fn = _make_scan_kernel_nopad(self.numel, num_experts, top_k)

        self._gather_fn = _make_gather_kernel_nopad(
            num_tokens, self.numel, hidden_size, self.dtype_str
        )

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"threads": _SCAN_THREADS}

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run moe_permute without padding.

        Args:
            hidden_states: [T, H] input activations.
            topk_ids: [T, K] int32 local expert assignments.

        Returns:
            perm_h:                    [T*K, H] tight hidden states.
            true_offsets:              [E] int32 tight start per local expert
            true_sizes:                [E] int32 true token count per local expert
            expert_first_token_offset: [E+1] int64 non-padded exclusive prefix-sum
            fwd_idx:                   [T*K] int32 forward mapping: flat_idx → tight slot
        """
        assert topk_ids.dtype == torch.int32
        assert hidden_states.is_cuda and topk_ids.is_cuda
        assert topk_ids.numel() == self.numel
        dev = hidden_states.device
        flat_ids = topk_ids.flatten().contiguous()
        E_local = self.num_experts

        expert_first_token_offset = torch.empty(E_local + 1, dtype=torch.int64, device=dev)
        true_offsets = torch.empty(E_local, dtype=torch.int32, device=dev)
        true_sizes = torch.empty(E_local, dtype=torch.int32, device=dev)
        permuted_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        fwd_idx = torch.empty(self.numel, dtype=torch.int32, device=dev)
        write_offsets = torch.empty(E_local, dtype=torch.int32, device=dev)

        threads = self.config["threads"]
        scan_fn = self._scan_fn(threads)
        scan_fn(
            flat_ids,
            expert_first_token_offset,
            true_offsets,
            true_sizes,
            permuted_idx,
            fwd_idx,
            write_offsets,
        )

        perm_h = torch.empty((self.numel, self.hidden_size), dtype=self.dtype, device=dev)
        gather_fn = self._gather_fn()
        gather_fn(hidden_states, permuted_idx, perm_h)

        return perm_h, true_offsets, true_sizes, expert_first_token_offset, fwd_idx
