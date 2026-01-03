import torch

from top.kernels.deepseek_nsa.mean_pooling_triton import mean_pooling
from top.kernels.deepseek_nsa.mean_pooling_triton import prepare_chunk_indices

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[3])  
def mean_pooling_tilelang_kernel(
    batch_size: int,
    total_seqlen: int,
    total_chunks: int,
    heads: int,
    dim: int,
    chunk_size: int,
    block_D: int = 64,
    threads: int = 128,
):
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        X_unpad: T.Tensor([total_seqlen, heads, dim], dtype),
        cu_seqlens: T.Tensor([batch_size + 1], T.int32),
        chunk_indices: T.Tensor([total_chunks, 2], T.int32),
        Output: T.Tensor([total_chunks, heads, dim], dtype),
    ):
        with T.Kernel(
            T.ceildiv(dim, block_D),
            total_chunks,
            heads,
            threads=threads
        ) as (i_d, i_t, i_h):
            accum = T.alloc_fragment([block_D], accum_dtype)
            d_start = i_d * block_D

            seq_id = chunk_indices[i_t, 0]
            local_chunk_id = chunk_indices[i_t, 1]
            start = cu_seqlens[seq_id]
            end = cu_seqlens[seq_id + 1]
            seqlen = end - start

            chunk_start = local_chunk_id * chunk_size
            chunk_end = T.min(chunk_start + chunk_size, seqlen)
            actual_bt = chunk_end - chunk_start

            for d in T.Parallel(block_D):
                accum[d] = T.cast(0, accum_dtype)
            for t_rel in T.serial(actual_bt):
                t_abs = start + chunk_start + t_rel
                for d in T.Parallel(block_D):
                    if d_start + d < dim:
                        accum[d] += T.cast(X_unpad[t_abs, i_h, d_start + d], accum_dtype)
            for d in T.Parallel(block_D):
                if d_start + d < dim:
                    Output[i_t, i_h, d_start + d] = T.cast(accum[d] / T.cast(actual_bt, accum_dtype), dtype)

    return main


def mean_pooling_tilelang(x_unpad, cu_seqlens, chunk_size, block_D=64):
    total_T, H, D = x_unpad.shape
    B = cu_seqlens.shape[0] - 1
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    total_chunks = chunk_indices.shape[0]

    kernel = mean_pooling_tilelang_kernel(
        batch_size=B,
        total_seqlen=total_T,
        total_chunks=total_chunks,
        heads=H,
        dim=D,
        chunk_size=chunk_size,
        block_D=block_D,
        threads=128,
    )
    return kernel(x_unpad, cu_seqlens, chunk_indices)


def test_varlen():
    print("=== 🌊 Testing Variable-Length Mode ===")
    device = "cuda"
    torch.manual_seed(42)

    seqlens = torch.tensor([100, 150], device=device, dtype=torch.int32)
    cu_seqlens = torch.zeros(seqlens.shape[0] + 1, device=device, dtype=torch.int32)
    cu_seqlens[1:] = seqlens.cumsum(0)
    total_T = cu_seqlens[-1].item()
    H, D, chunk_size = 4, 64, 32

    x_unpad = torch.randn(total_T, H, D, dtype=torch.float16, device=device)
    # x_triton = x_unpad.unsqueeze(0)  # (1, total_T, H, D)

    # Triton
    out_triton = mean_pooling(x_unpad.unsqueeze(0), chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    out_triton = out_triton.squeeze(0)

    # TileLang
    out_tilelang = mean_pooling_tilelang(x_unpad, cu_seqlens, chunk_size)

    print(f"Triton:  {out_triton.shape}")
    print(f"TileLang: {out_tilelang.shape}")
    print(f"Max diff: {(out_triton - out_tilelang).abs().max().item():.6f}")
    torch.testing.assert_close(out_triton.float(), out_tilelang.float(), atol=1e-2, rtol=1e-2)
    print("✅ Varlen test passed!\n")


# Test 2: Fixed-Length
def test_fixed():
    print("=== 📏 Testing Fixed-Length Mode ===")
    device = "cuda"
    torch.manual_seed(42)

    B, T, H, D = 3, 1024, 128, 128
    chunk_size = 32

    x = torch.randn(B, T, H, D, dtype=torch.float16, device=device)
    out_triton = mean_pooling(x, chunk_size=chunk_size, cu_seqlens=None, head_first=False)  # (B, NT, H, D)
    out_triton_reshaped = out_triton.view(-1, H, D)  # (B*NT, H, D)

    x_unpad = x.view(-1, H, D)  # (B*T, H, D)
    cu_seqlens = torch.arange(0, (B + 1) * T, T, dtype=torch.int32, device=device)  # [0, T, 2T]
    out_tilelang = mean_pooling_tilelang(x_unpad, cu_seqlens, chunk_size)  # (total_chunks, H, D)

    print(f"Triton:  {out_triton_reshaped.shape}")
    print(f"TileLang: {out_tilelang.shape}")
    print(f"Max diff: {(out_triton_reshaped - out_tilelang).abs().max().item():.6f}")
    torch.testing.assert_close(out_triton_reshaped.float(), out_tilelang.float(), atol=1e-2, rtol=1e-2)
    print("✅ Fixed-length test passed!\n")


if __name__ == "__main__":
    test_varlen()
    test_fixed()
    print("🎉 All tests passed! TileLang and Triton outputs match perfectly.")