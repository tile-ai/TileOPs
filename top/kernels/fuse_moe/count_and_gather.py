from typing import Optional
import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel


def _count_seq_and_cuseq_kernel(total_num_topk: int, num_expert: int, start_expert: int,
                                end_expert: int, tile_m: int):

    @tilelang.jit(out_idx=[1, 2, 3, 4])
    def _count_seq_and_cuseq_fwd_func(block_size: int = 256):

        @T.prim_func
        def _count_seq_and_cuseq_main(
            topk_ids_flat: T.Tensor[(total_num_topk,), T.int32],
            seqlens: T.Tensor[(num_expert,), T.int32],
            cu_seqlens: T.Tensor[(num_expert + 1,), T.int32],
            tiles: T.Tensor[(num_expert,), T.int32],
            topk_pos: T.Tensor[(total_num_topk,), T.int32],
        ):
            with T.Kernel(1, threads=block_size) as bx:
                tx = T.get_thread_binding()
                seqlens_shm = T.alloc_shared((num_expert,), T.int32)

                for i in T.serial(T.ceildiv(num_expert, block_size)):
                    iexpert = i * block_size + tx
                    if iexpert < num_expert:
                        seqlens_shm[iexpert] = 0

                for i in T.serial(T.ceildiv(total_num_topk, block_size)):
                    idx = i * block_size + tx
                    if idx < total_num_topk:
                        iexpert = topk_ids_flat[idx]
                        if iexpert >= start_expert and iexpert < end_expert:
                            T.atomic_add(seqlens_shm[iexpert - start_expert], 1)
                        topk_pos[idx] = -1

                T.sync_threads()

                if tx == 0:
                    cu_seqlens[0] = 0
                    for i in T.serial(num_expert):
                        iseq = seqlens_shm[i]
                        seqlens[i] = iseq
                        tiles[i] = (iseq + tile_m - 1) // tile_m
                        cu_seqlens[i + 1] = cu_seqlens[i] + iseq

        return _count_seq_and_cuseq_main

    return _count_seq_and_cuseq_fwd_func


def _gather_kernel(num_seq: int, hidden_size: int, num_topk: int, total_num_topk: int,
                   num_expert: int, start_expert: int, end_expert: int):

    @tilelang.jit(out_idx=[3, 4, 5])
    def _gather_fwd_func(block_size: int = 256):

        @T.prim_func
        def _gather_main(
            x: T.Tensor[(num_seq, hidden_size), T.float32],
            topk_ids_flat: T.Tensor[(total_num_topk,), T.int32],
            cu_seqlens: T.Tensor[(num_expert + 1,), T.int32],
            seqlens_runtime: T.Tensor[(num_expert,), T.int32],
            topk_pos: T.Tensor[(total_num_topk,), T.int32],
            gate_up_input_full: T.Tensor[(total_num_topk, hidden_size), T.float32],
        ):
            with T.Kernel(1, threads=block_size) as bx:
                tx = T.get_thread_binding()

                for i in T.serial(T.ceildiv(num_expert, block_size)):
                    iexpert = i * block_size + tx
                    if iexpert < num_expert:
                        seqlens_runtime[iexpert] = 0

                for i in T.serial(T.ceildiv(total_num_topk, block_size)):
                    idx = i * block_size + tx
                    if idx < total_num_topk:
                        topk_pos[idx] = -1
                        iexpert = topk_ids_flat[idx]
                        if iexpert >= start_expert and iexpert < end_expert:
                            expert_idx = iexpert - start_expert
                            pos_in_expert = T.atomic_add(seqlens_runtime[expert_idx],
                                                         1,
                                                         return_prev=True)
                            irow = cu_seqlens[expert_idx] + pos_in_expert
                            topk_pos[idx] = irow
                            iseq = idx // num_topk
                            for ih in T.serial(hidden_size):
                                gate_up_input_full[irow, ih] = x[iseq, ih]

        return _gather_main

    return _gather_fwd_func


class CountAndGatherKernel(Kernel):
    """Count and gather kernel for MoE."""

    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 num_seq: int,
                 hidden_size: int,
                 num_topk: int,
                 num_expert: int,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.num_seq = num_seq
        self.hidden_size = hidden_size
        self.num_topk = num_topk
        self.num_expert = num_expert

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_size": 256,
            "tile_m": 16
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_sizes = [256, 512]
        tile_ms = [16, 32]
        configs = []
        for block_size in block_sizes:
            for tile_m in tile_ms:
                configs.append({
                    "block_size": block_size,
                    "tile_m": tile_m
                })
        return configs

    def forward(self, x, topk_ids, rank_ep=0):
        """Run the kernel
        
        Args:
            x: Input token features [num_seq, hidden_size]
            topk_ids: Expert assignment for each token [num_seq, num_topk]
            rank_ep: Expert parallel rank
            
        Returns:
            gate_up_input: Gathered input for gate and up projection
            topk_pos: Position mapping for each token
            seqlens: Number of tokens per expert
            cu_seqlens: Cumulative sequence lengths
            tiles: Number of tiles per expert
        """
        return self.count_and_gather(x, topk_ids, rank_ep)

    def count_and_gather(self, x, topk_ids, rank_ep=0):
        """Count and gather tokens by expert.
        
        Args:
            x: Input token features [num_seq, hidden_size]
            topk_ids: Expert assignment for each token [num_seq, num_topk]
            rank_ep: Expert parallel rank
            
        Returns:
            gate_up_input: Gathered input for gate and up projection
            topk_pos: Position mapping for each token
            seqlens: Number of tokens per expert
            cu_seqlens: Cumulative sequence lengths
            tiles: Number of tiles per expert
        """
        num_seq, hidden_size = x.shape
        num_topk = topk_ids.shape[1]
        total_num_topk = num_seq * num_topk
        num_expert = self.num_expert
        
        start_expert = rank_ep * num_expert
        end_expert = (rank_ep + 1) * num_expert
        
        block_size = self.config["block_size"]

        topk_ids_flat = topk_ids.reshape(-1).contiguous()

        seqlens, cu_seqlens, tiles, _ = _count_seq_and_cuseq_kernel(
            total_num_topk=total_num_topk,
            num_expert=num_expert,
            start_expert=start_expert,
            end_expert=end_expert,
            tile_m=self.config["tile_m"],
        )(block_size)(topk_ids_flat)

        x_fp32 = x.to(torch.float32)
        seqlens_runtime, topk_pos, gate_up_input_full = _gather_kernel(
            num_seq=num_seq,
            hidden_size=hidden_size,
            num_topk=num_topk,
            total_num_topk=total_num_topk,
            num_expert=num_expert,
            start_expert=start_expert,
            end_expert=end_expert,
        )(block_size)(x_fp32, topk_ids_flat, cu_seqlens)

        total_tokens = cu_seqlens[-1].item()
        gate_up_input = gate_up_input_full[:total_tokens].to(x.dtype)

        return gate_up_input, topk_pos, seqlens_runtime, cu_seqlens, tiles

