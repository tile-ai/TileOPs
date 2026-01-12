import torch

from top.ops import GroupedGemmNNOp, GroupedGemmNTOp, GroupedGemmTNOp

from .function import Function

__all__ = ['GroupedGemmFunc']


# flake8: noqa
class GroupedGemmCtx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B, batch_sizes, batch_offsets, batch_padded_offsets, fwd_op, dA_op, dB_op):
        output = fwd_op(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
        ctx.save_for_backward(A, B, batch_sizes, batch_offsets, batch_padded_offsets)
        ctx.dA_op = dA_op
        ctx.dB_op = dB_op
        return output  # noqa: R504

    @staticmethod
    def backward(ctx, grad_output):
        A_matrix, B_matrix, batch_sizes, batch_offsets, batch_padded_offsets = ctx.saved_tensors

        def maybe_contiguous(x):
            if x.stride(-1) != 1:
                return x.contiguous()
            return x

        A_matrix, B_matrix, grad_output, batch_sizes, batch_offsets, batch_padded_offsets = [
            maybe_contiguous(x) for x in (A_matrix, B_matrix, grad_output, batch_sizes,
                                          batch_offsets, batch_padded_offsets)
        ]
        dA = ctx.dA_op(grad_output, B_matrix, batch_sizes, batch_offsets, batch_padded_offsets)
        dB = ctx.dB_op(grad_output, A_matrix, batch_sizes, batch_offsets, batch_padded_offsets)
        return dA, dB, None, None, None, None, None, None


class GroupedGemmFunc(Function):

    def __init__(self, batch_sum, batch_count, N, K, dtype=torch.float16, tune=False):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype

        self.fwd_op = GroupedGemmNTOp(batch_sum, batch_count, N, K, dtype, tune=tune)
        self.dA_op = GroupedGemmNNOp(batch_sum, batch_count, K, N, dtype, tune=tune)
        self.dB_op = GroupedGemmTNOp(batch_sum, batch_count, N, K, dtype, tune=tune)

    def forward(self, A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor) -> torch.Tensor:
        return GroupedGemmCtx.apply(A, B, batch_sizes, batch_offsets, batch_padded_offsets,
                                    self.fwd_op, self.dA_op, self.dB_op)
