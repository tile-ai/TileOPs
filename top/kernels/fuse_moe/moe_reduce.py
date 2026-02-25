import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel


def _moe_reduce_kernel(total_num_seq: int, num_seq: int, hidden_size: int, num_topk: int):

    @tilelang.jit(out_idx=[4])
    def _reduce_fwd_func(block_size: int = 256):

        @T.prim_func
        def _reduce_main(
            x_fp32: T.Tensor[(total_num_seq, hidden_size), T.float32],
            topk_pos: T.Tensor[(num_seq, num_topk), T.int32],
            topk_scale_fp32: T.Tensor[(num_seq, num_topk), T.float32],
            shared_output_fp32: T.Tensor[(num_seq, hidden_size), T.float32],
            output_fp32: T.Tensor[(num_seq, hidden_size), T.float32],
        ):
            with T.Kernel(num_seq, threads=block_size) as by:
                tx = T.get_thread_binding()
                for ih_blk in T.serial(T.ceildiv(hidden_size, block_size)):
                    ih = ih_blk * block_size + tx
                    if ih < hidden_size:
                        acc = T.alloc_var(T.float32)
                        acc = 0.0
                        for itopk in T.serial(num_topk):
                            ipos = topk_pos[by, itopk]
                            if ipos >= 0 and ipos < total_num_seq:
                                acc += x_fp32[ipos, ih] * topk_scale_fp32[by, itopk]
                        acc += shared_output_fp32[by, ih]
                        output_fp32[by, ih] = acc

        return _reduce_main

    return _reduce_fwd_func


class MoeReduceKernel(Kernel):
    """Reduce kernel for MoE.

    Performs scatter-add aggregation operation, aggregating expert outputs back to original positions.
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 num_seq: int,
                 hidden_size: int,
                 num_topk: int,
                 config: dict = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.num_seq = num_seq
        self.hidden_size = hidden_size
        self.num_topk = num_topk

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_size": 256,
            "items_per_16b": 8  # For FP16
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_sizes = [256, 512]
        configs = []
        for block_size in block_sizes:
            configs.append({"block_size": block_size, "items_per_16b": 8})
        return configs

    def forward(self, x, topk_pos, topk_scale, shared_output=None):
        """Forward pass of ReduceKernel.

        Args:
            x: Input tensor (expert outputs) of shape [num_tokens, hidden_size]
            topk_pos: Position mapping tensor of shape [num_seq, num_topk]
            topk_scale: Weight tensor of shape [num_seq, num_topk]
            shared_output: Optional shared output tensor of shape [num_seq, hidden_size]

        Returns:
            output: Aggregated output tensor of shape [num_seq, hidden_size]
        """
        num_seq, num_topk = topk_pos.shape
        num_tokens, hidden_size = x.shape
        block_size = self.config["block_size"]

        if num_topk == 0:
            if shared_output is None:
                return torch.zeros((num_seq, hidden_size), dtype=x.dtype, device=x.device)
            return shared_output.to(dtype=x.dtype)

        x_fp32 = x.to(torch.float32).contiguous()
        topk_scale_fp32 = topk_scale.to(torch.float32).contiguous()
        topk_pos_i32 = topk_pos.to(torch.int32).contiguous()

        if shared_output is None:
            shared_output_fp32 = torch.zeros((num_seq, hidden_size),
                                             dtype=torch.float32,
                                             device=x.device)
        else:
            shared_output_fp32 = shared_output.to(torch.float32).contiguous()

        output_fp32 = _moe_reduce_kernel(
            total_num_seq=num_tokens,
            num_seq=num_seq,
            hidden_size=hidden_size,
            num_topk=num_topk,
        )(block_size)(x_fp32, topk_pos_i32, topk_scale_fp32, shared_output_fp32)

        return output_fp32.to(x.dtype)
