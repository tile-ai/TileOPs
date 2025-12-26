import torch
from torch import nn
from top.functions import NativeSparseAttentionFunc

from top.kernels.deepseek_nsa.nsa_torch import naive_nsa


class NativeSparseAttentionLayer(nn.Module):

    def __init__(
        self,
        batch,
        heads,
        seq_len,
        dim,
        is_causal,
        scale=None,
        block_size=64,
        groups=1,
        selected_blocks=16,
        tune=False
    ):
        super().__init__()

        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.scale = scale
        self.block_size = block_size
        self.groups = groups
        self.selected_blocks = selected_blocks
        self.tune = tune

        self.fn = NativeSparseAttentionFunc(
            batch, heads, seq_len, dim, is_causal, scale, block_size, groups, selected_blocks, tune=tune)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor) -> torch.Tensor:
        return self.fn(Q, K, V, BlockIndices)


def main():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1

    block_T = min(128, 16)

    kernel = NativeSparseAttentionLayer(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
        scale=scale,
        tune=True,
    )


    torch.random.manual_seed(0)
    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda").requires_grad_(True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device="cuda").requires_grad_(True)
    g_slc = torch.ones((B, SEQ_LEN, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    g_swa = torch.ones((B, SEQ_LEN, HQ), dtype=dtype, device="cuda").requires_grad_(True)
    DO = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device="cuda")

    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.long, device="cuda")
    block_counts = torch.zeros((B, SEQ_LEN, H), dtype=torch.long, device="cuda")
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
                block_counts[b, t, h] = (block_indices[b, t, h] != SEQ_LEN).sum().item()
    block_indices = block_indices.sort(-1)[0]

    out = kernel.forward(Q, K, V, block_indices.to(torch.int32))

    ref = naive_nsa(
        q=Q,
        k=K,
        v=V,
        g_slc=g_slc,
        g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
    )

    print("out", out)
    print("ref", ref)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()