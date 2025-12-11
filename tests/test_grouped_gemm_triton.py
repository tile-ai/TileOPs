import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import triton
import triton.language as tl
import time
import argparse
import math


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'NUM_SM': 128}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_gemm_nt_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_tiles = tl.cdiv(gm, BLOCK_SIZE_M) * tl.cdiv(gn, BLOCK_SIZE_N)

        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m = tile_idx_in_gemm // tl.cdiv(gn, BLOCK_SIZE_N)
            tile_n = tile_idx_in_gemm % tl.cdiv(gn, BLOCK_SIZE_N)

            rm = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            rk = tl.arange(0, BLOCK_SIZE_K)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
                a = tl.load(
                    a_ptr + rm[:, None] * lda + rk[None, :],
                    mask=(rm[:, None] < gm) & (rk[None, :] < gk),
                    other=0.0)
                b = tl.load(
                    b_ptr + rk[:, None] * ldb + rn[None, :],
                    mask=(rk[:, None] < gk) & (rn[None, :] < gn),
                    other=0.0)
                acc += tl.dot(a, b)
                rk += BLOCK_SIZE_K

            tl.store(
                c_ptr + rm[:, None] * ldc + rn[None, :],
                acc.to(tl.float16),
                mask=(rm[:, None] < gm) & (rn[None, :] < gn))
            tile_idx += NUM_SM

        last_problem_end += num_tiles


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'NUM_SM': 128}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_gemm_nn_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_tiles_m = tl.cdiv(gm, BLOCK_SIZE_M)
        num_tiles_k = tl.cdiv(gk, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_k

        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m = tile_idx_in_gemm // num_tiles_k
            tile_k = tile_idx_in_gemm % num_tiles_k
            rm = tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rk = tile_k * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for n in range(0, tl.cdiv(gn, BLOCK_SIZE_K)):
                rn = n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                a = tl.load(
                    a_ptr + rm[:, None] * lda + rn[None, :],
                    mask=(rm[:, None] < gm) & (rn[None, :] < gn),
                    other=0.0)
                b = tl.load(
                    b_ptr + rn[:, None] * ldb + rk[None, :],
                    mask=(rn[:, None] < gn) & (rk[None, :] < gk),
                    other=0.0)
                acc += tl.dot(a, b)

            tl.store(
                c_ptr + rm[:, None] * ldc + rk[None, :],
                acc.to(tl.float16),
                mask=(rm[:, None] < gm) & (rk[None, :] < gk))
            tile_idx += NUM_SM

        last_problem_end += num_tiles


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'NUM_SM': 128}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_gemm_tn_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gk = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gm = tl.load(group_gemm_sizes + g * 3 + 2)

        num_tiles_k = tl.cdiv(gk, BLOCK_SIZE_M)
        num_tiles_n = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_tiles_k * num_tiles_n

        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.float16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.float16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.float16))
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_k = tile_idx_in_gemm // num_tiles_n
            tile_n = tile_idx_in_gemm % num_tiles_n
            rk = tile_k * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            rn = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for m_block in range(0, tl.cdiv(gm, BLOCK_SIZE_K)):
                rm = m_block * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                a = tl.load(
                    a_ptr + rk[:, None] * lda + rm[None, :],
                    mask=(rk[:, None] < gk) & (rm[None, :] < gm),
                    other=0.0)
                b = tl.load(
                    b_ptr + rm[:, None] * ldb + rn[None, :],
                    mask=(rm[:, None] < gm) & (rn[None, :] < gn),
                    other=0.0)
                acc += tl.dot(a, b)

            tl.store(
                c_ptr + rk[:, None] * ldc + rn[None, :],
                acc.to(tl.float16),
                mask=(rk[:, None] < gk) & (rn[None, :] < gn))
            tile_idx += NUM_SM

        last_problem_end += num_tiles


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 256,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128
        }),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'NUM_SM': 128}),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256, 'NUM_SM': 128}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_gemm_tt_kernel(
    group_dO_ptrs,
    group_A_ptrs,
    group_C_ptrs,
    group_gemm_sizes,
    g_lds,
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    last_problem_end = 0

    # loop over groups
    for g in range(group_size):

        M = tl.load(group_gemm_sizes + g * 3)  # reduction dim
        N = tl.load(group_gemm_sizes + g * 3 + 1)  # output rows
        K = tl.load(group_gemm_sizes + g * 3 + 2)  # output cols

        # output tiles: [N, K]
        num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)  # tile rows
        num_tiles_k = tl.cdiv(K, BLOCK_SIZE_K)  # tile cols
        num_tiles = num_tiles_n * num_tiles_k

        # check if pid falls into this group
        while (pid >= last_problem_end) and (pid < last_problem_end + num_tiles):

            # lds
            ld_dO = tl.load(g_lds + g * 3)
            ld_A = tl.load(g_lds + g * 3 + 1)
            ld_C = tl.load(g_lds + g * 3 + 2)

            # base pointers
            dO_ptr = tl.load(group_dO_ptrs + g).to(tl.pointer_type(tl.float16))
            A_ptr = tl.load(group_A_ptrs + g).to(tl.pointer_type(tl.float16))
            C_ptr = tl.load(group_C_ptrs + g).to(tl.pointer_type(tl.float16))

            tile_idx = pid - last_problem_end
            tile_n = tile_idx // num_tiles_k
            tile_k = tile_idx % num_tiles_k

            rn = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # rows of C
            rk = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)  # cols of C

            # accumulator [N, K]
            acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)

            # reduce over M in blocks
            for m_block in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
                rm = m_block * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

                # dO_block shape: [BLOCK_M, BLOCK_N]
                dO_block = tl.load(
                    dO_ptr + rm[:, None] * ld_dO + rn[None, :],
                    mask=(rm[:, None] < M) & (rn[None, :] < N),
                    other=0.0,
                )

                # A_block shape: [BLOCK_M, BLOCK_K]
                A_block = tl.load(
                    A_ptr + rm[:, None] * ld_A + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K),
                    other=0.0,
                )

                # (dOᵀ @ A) partial accumulate
                acc += tl.dot(tl.trans(dO_block), A_block)

            # store result
            tl.store(
                C_ptr + rn[:, None] * ld_C + rk[None, :],
                acc.to(tl.float16),
                mask=(rn[:, None] < N) & (rk[None, :] < K))

            pid += NUM_SM

        last_problem_end += num_tiles


def prepare_nt_inputs(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes_list = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes_list[i] += 1

    padding_M = 128
    device = 'cuda'
    batch_sum_calc = sum(batch_sizes_list)
    batch_count_calc = len(batch_sizes_list)

    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count_calc - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count_calc - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)
    A = torch.randn(batch_sum_calc, K, device=device, dtype=dtype)
    B = torch.randn(batch_count_calc, K, N, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)

    return A, B, batch_sizes, batch_offsets, batch_padded_offsets


def prepare_nn_inputs(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes_list = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes_list[i] += 1

    padding_M = 128
    device = 'cuda'
    batch_sum_calc = sum(batch_sizes_list)
    batch_count_calc = len(batch_sizes_list)

    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count_calc - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count_calc - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)
    A = torch.randn(batch_sum_calc, N, device=device, dtype=dtype)
    B = torch.randn(batch_count_calc, N, K, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)

    return A, B, batch_sizes, batch_offsets, batch_padded_offsets


def prepare_tn_inputs(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes_list = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes_list[i] += 1

    padding_M = 128
    device = 'cuda'
    batch_sum_calc = sum(batch_sizes_list)
    batch_count_calc = len(batch_sizes_list)

    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count_calc - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count_calc - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)
    A = torch.randn(N, batch_sum_calc, device=device, dtype=dtype)
    B = torch.randn(batch_sum_calc, K, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)

    return A, B, batch_sizes, batch_offsets, batch_padded_offsets


def prepare_tt_inputs(batch_sum: int, batch_count: int, N: int, K: int, dtype=torch.float16):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes_list = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes_list[i] += 1

    padding_M = 128
    device = 'cuda'
    batch_sum_calc = sum(batch_sizes_list)
    batch_count_calc = len(batch_sizes_list)

    batch_offsets_list = [0]
    batch_padded_offsets_list = [0]
    for i in range(batch_count_calc - 1):
        batch_offsets_list.append(batch_offsets_list[-1] + batch_sizes_list[i])
    for i in range(batch_count_calc - 1):
        batch_padded_offsets_list.append(batch_padded_offsets_list[-1] +
                                         math.ceil((batch_sizes_list[i] + 1) / padding_M) *
                                         padding_M)
    dO = torch.randn(batch_sum_calc, N, device=device, dtype=dtype)
    A = torch.randn(batch_sum_calc, K, device=device, dtype=dtype)
    batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
    batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
    batch_padded_offsets = torch.tensor(batch_padded_offsets_list, device=device, dtype=torch.int32)

    return dO, A, batch_sizes, batch_offsets, batch_padded_offsets


def prepare_grouped_tensors_nt(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor):
    batch_sizes_list = batch_sizes.tolist()
    group_size = len(batch_sizes_list)

    A_addrs, B_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []

    start = 0
    for i, size in enumerate(batch_sizes_list):
        end = start + size
        M, K = size, A.shape[1]
        _, N = B.shape[1], B.shape[2]

        A_group = A[start:end]
        B_group = B[i]
        C_group = torch.empty((M, N), device=A.device, dtype=A.dtype)

        A_addrs.append(A_group.data_ptr())
        B_addrs.append(B_group.data_ptr())
        C_addrs.append(C_group.data_ptr())
        g_sizes.extend([M, N, K])
        g_lds.extend([A_group.stride(0), B_group.stride(0), C_group.stride(0)])

        group_C.append(C_group)
        start = end

    return (torch.tensor(A_addrs, device=A.device), torch.tensor(B_addrs, device=B.device),
            torch.tensor(C_addrs, device=A.device),
            torch.tensor(g_sizes, dtype=torch.int32,
                         device=A.device), torch.tensor(g_lds, dtype=torch.int32,
                                                        device=A.device), group_C)


def prepare_grouped_tensors_nn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor):
    batch_sizes_list = batch_sizes.tolist()
    group_size = len(batch_sizes_list)

    A_addrs, B_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []

    start = 0
    for i, size in enumerate(batch_sizes_list):
        end = start + size
        M = size
        N = A.shape[1]
        K = B.shape[2]
        assert B.shape[
            1] == N, f"B.shape[1]={B.shape[1]} != N={N}, B should have shape (batch_count, N, K)"

        A_group = A[start:end]
        B_group = B[i]
        C_group = torch.empty((M, K), device=A.device, dtype=A.dtype)  # dA: (M, K)

        A_addrs.append(A_group.data_ptr())
        B_addrs.append(B_group.data_ptr())
        C_addrs.append(C_group.data_ptr())
        g_sizes.extend([M, N, K])
        g_lds.extend([A_group.stride(0), B_group.stride(0), C_group.stride(0)])
        group_C.append(C_group)
        start = end

    return (torch.tensor(A_addrs, device=A.device), torch.tensor(B_addrs, device=B.device),
            torch.tensor(C_addrs, device=A.device),
            torch.tensor(g_sizes, dtype=torch.int32,
                         device=A.device), torch.tensor(g_lds, dtype=torch.int32,
                                                        device=A.device), group_C)


def prepare_grouped_tensors_tn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor):
    batch_sizes_list = batch_sizes.tolist()
    group_size = len(batch_sizes_list)

    A_addrs, B_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []

    start = 0
    for i, size in enumerate(batch_sizes_list):
        end = start + size
        K, M = A.shape[0], size
        M_b, N = size, B.shape[1]

        A_group = A[:, start:end]
        B_group = B[start:end]
        C_group = torch.empty((K, N), device=A.device, dtype=A.dtype)

        A_addrs.append(A_group.data_ptr())
        B_addrs.append(B_group.data_ptr())
        C_addrs.append(C_group.data_ptr())
        g_sizes.extend([K, N, M])
        g_lds.extend([A_group.stride(0), B_group.stride(0), C_group.stride(0)])

        group_C.append(C_group)
        start = end

    return (torch.tensor(A_addrs, device=A.device), torch.tensor(B_addrs, device=B.device),
            torch.tensor(C_addrs, device=A.device),
            torch.tensor(g_sizes, dtype=torch.int32,
                         device=A.device), torch.tensor(g_lds, dtype=torch.int32,
                                                        device=A.device), group_C)


def prepare_grouped_tensors_tt(dO: torch.Tensor, A: torch.Tensor, batch_sizes: torch.Tensor):
    batch_sizes_list = batch_sizes.tolist()
    group_size = len(batch_sizes_list)

    dO_addrs, A_addrs, C_addrs = [], [], []
    g_sizes, g_lds = [], []
    group_C = []

    start = 0
    for i, size in enumerate(batch_sizes_list):
        end = start + size
        M = size
        N = dO.shape[1]
        K = A.shape[1]

        dO_group = dO[start:end]
        A_group = A[start:end]
        C_group = torch.empty((N, K), device=dO.device, dtype=dO.dtype)

        dO_addrs.append(dO_group.data_ptr())
        A_addrs.append(A_group.data_ptr())
        C_addrs.append(C_group.data_ptr())
        g_sizes.extend([M, N, K])
        g_lds.extend([dO_group.stride(0), A_group.stride(0), C_group.stride(0)])
        group_C.append(C_group)
        start = end

    return (torch.tensor(dO_addrs, device=dO.device), torch.tensor(A_addrs, device=A.device),
            torch.tensor(C_addrs, device=dO.device),
            torch.tensor(g_sizes, dtype=torch.int32, device=dO.device),
            torch.tensor(g_lds, dtype=torch.int32, device=dO.device), group_C)


def grouped_gemm_nt(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
    (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds,
     group_C) = prepare_grouped_tensors_nt(A, B, batch_sizes)

    grid = lambda META: (META['NUM_SM'],)
    grouped_gemm_nt_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(batch_sizes))

    return torch.cat(group_C, dim=0)


def grouped_gemm_nn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
    (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds,
     group_C) = prepare_grouped_tensors_nn(A, B, batch_sizes)

    grid = lambda META: (META['NUM_SM'],)
    grouped_gemm_nn_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(batch_sizes))

    return torch.cat(group_C, dim=0)


def grouped_gemm_tn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
    (d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds,
     group_C) = prepare_grouped_tensors_tn(A, B, batch_sizes)

    grid = lambda META: (META['NUM_SM'],)
    grouped_gemm_tn_kernel[grid](d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, len(batch_sizes))

    return torch.stack(group_C, dim=0)


def grouped_gemm_tt(dO: torch.Tensor, A: torch.Tensor, batch_sizes: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    batch_padded_offsets: torch.Tensor) -> torch.Tensor:
    (dO_addrs, A_addrs, C_addrs, g_sizes, g_lds,
     group_C) = prepare_grouped_tensors_tt(dO, A, batch_sizes)

    grid = lambda META: (META['NUM_SM'],)
    grouped_gemm_tt_kernel[grid](dO_addrs, A_addrs, C_addrs, g_sizes, g_lds, len(batch_sizes))

    return torch.stack(group_C, dim=0)


def ref_program_nt(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                   batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
    output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        output[start:end] = torch.mm(A[start:end], B[i])
        start = end
    return output


def ref_program_nn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                   batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
    total_batch = int(batch_sizes.sum().item())
    assert A.shape[0] == total_batch, f"A.shape[0]={A.shape[0]} != sum(batch_sizes)={total_batch}"
    assert B.shape[0] == len(
        batch_sizes), f"B.shape[0]={B.shape[0]} != len(batch_sizes)={len(batch_sizes)}"
    output = torch.empty((total_batch, B.shape[2]), device=A.device, dtype=A.dtype)
    start = 0
    for i, size in enumerate(batch_sizes):
        size = int(size.item())
        end = start + size
        output[start:end] = torch.mm(A[start:end], B[i])
        start = end
    return output


def ref_program_tn(A: torch.Tensor, B: torch.Tensor, batch_sizes: torch.Tensor,
                   batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
    output = torch.zeros((len(batch_sizes), A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)
    start = 0
    for i, size in enumerate(batch_sizes):
        end = start + size
        output[i] = torch.mm(A[:, start:end], B[start:end, :])
        start = end
    return output


def ref_program_tt(dO: torch.Tensor, A: torch.Tensor, batch_sizes: torch.Tensor,
                   batch_offsets: torch.Tensor, batch_padded_offsets: torch.Tensor):
    output = torch.zeros((len(batch_sizes), dO.shape[1], A.shape[1]),
                         device=dO.device,
                         dtype=dO.dtype)
    start = 0
    for i, size in enumerate(batch_sizes):
        size_val = size.item()
        end = start + size_val
        group_dO = dO[start:end]
        group_A = A[start:end]
        output[i] = torch.mm(group_dO.T, group_A)
        start = end
    return output


def check_kernel(kernel_func, ref_func, inputs, kernel_name):
    print(f"\n=== Checking {kernel_name} ===\n")
    outputs = kernel_func(*inputs)
    outputs_ref = ref_func(*inputs)

    print("Output data:")
    print("  Output 0:")
    print(f"    shape: {outputs.shape} vs {outputs_ref.shape}")
    print(f"    dtype: {outputs.dtype} vs {outputs_ref.dtype}")

    if torch.allclose(outputs, outputs_ref, atol=1e+1, rtol=1e-1):
        print(f"All checks passed for {kernel_name}.")
        return True
    else:
        diff = torch.abs(outputs - outputs_ref)
        print(f"最大差异: {diff.max().item()}")
        print(f"平均差异: {diff.mean().item()}")
        print(f"差异大于1e-1的元素数量: {(diff > 1e-1).sum().item()}")
        print(f"输出范围: [{outputs.min().item()}, {outputs.max().item()}]")
        print(f"参考范围: [{outputs_ref.min().item()}, {outputs_ref.max().item()}]")

        # 查看前几个值的差异
        for i in range(min(5, len(outputs))):
            print(f"输出[{i}]: {outputs[i][:5].tolist()}")
            print(f"参考[{i}]: {outputs_ref[i][:5].tolist()}")
            print(f"差异[{i}]: {(outputs[i][:5] - outputs_ref[i][:5]).tolist()}")
        print(f"Checks failed for {kernel_name}.")
        return False


def profile_kernel(kernel_func, inputs, kernel_name, total_flops):
    print(f"\n===== Profiling {kernel_name} =====")

    for _ in range(5):
        _ = kernel_func(*inputs)
    torch.cuda.synchronize()

    start_time = time.time()
    for _ in range(10):
        _ = kernel_func(*inputs)
    torch.cuda.synchronize()
    elapsed_time = (time.time() - start_time) / 10

    tflops = (total_flops / 1e12) / elapsed_time
    bandwidth = 0.01

    print(f"{kernel_name} latency: {elapsed_time * 1000:.2f} ms")
    print(f"{kernel_name} TFlops: {tflops:.2f} TFlops")
    print(f"{kernel_name} Bandwidth: {bandwidth:.2f} GB/s")


def calculate_flops_nt(batch_sizes, K, N):
    return 2.0 * sum(size * K * N for size in batch_sizes)


def calculate_flops_nn(batch_sizes, K, N):
    return 2.0 * sum(size * N * K for size in batch_sizes)


def calculate_flops_tn(batch_sizes, K, N):
    return 2.0 * sum(size * N * K for size in batch_sizes)


def calculate_flops_tt(batch_sizes, K, N):
    return 2.0 * sum(size * N * K for size in batch_sizes)


def test_grouped_gemm_nt(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    print("Testing grouped_gemm_nt (forward)...")

    inputs = prepare_nt_inputs(batch_sum, batch_count, K, N, dtype)
    batch_sizes = inputs[2]

    success = check_kernel(grouped_gemm_nt, ref_program_nt, inputs, "grouped_gemm_nt")

    if success:
        total_flops = calculate_flops_nt(batch_sizes.tolist(), K, N)
        profile_kernel(grouped_gemm_nt, inputs, "grouped_gemm_nt", total_flops)

    return success


def test_grouped_gemm_nn(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    print("\nTesting grouped_gemm_nn (backward dA)...")

    inputs = prepare_nn_inputs(batch_sum, batch_count, K, N, dtype)
    print("Generated inputs:")
    print(f"  A: {inputs[0].shape}, requires_grad: False")
    print(f"  B: {inputs[1].shape}, requires_grad: False")
    print(f"  batch_sizes: {inputs[2]}")

    batch_sizes = inputs[2]
    success = check_kernel(grouped_gemm_nn, ref_program_nn, inputs, "grouped_gemm_nn")

    if success:
        total_flops = calculate_flops_nn(batch_sizes.tolist(), K, N)
        profile_kernel(grouped_gemm_nn, inputs, "grouped_gemm_nn", total_flops)

    return success


def test_grouped_gemm_tn(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    print("\nTesting grouped_gemm_tn (backward dB)...")

    inputs = prepare_tn_inputs(batch_sum, batch_count, K, N, dtype)
    batch_sizes = inputs[2]

    success = check_kernel(grouped_gemm_tn, ref_program_tn, inputs, "grouped_gemm_tn")

    if success:
        total_flops = calculate_flops_tn(batch_sizes.tolist(), K, N)
        profile_kernel(grouped_gemm_tn, inputs, "grouped_gemm_tn", total_flops)

    return success


def test_grouped_gemm_tt(batch_sum: int, batch_count: int, K: int, N: int, dtype=torch.float16):
    print("\nTesting grouped_gemm_tn (backward dB)...")

    inputs = prepare_tt_inputs(batch_sum, batch_count, K, N, dtype)
    batch_sizes = inputs[2]

    success = check_kernel(grouped_gemm_tt, ref_program_tt, inputs, "grouped_gemm_tt")

    if success:
        total_flops = calculate_flops_tt(batch_sizes.tolist(), K, N)
        profile_kernel(grouped_gemm_tt, inputs, "grouped_gemm_tt", total_flops)

    return success


def main():
    parser = argparse.ArgumentParser(description='Triton Grouped GEMM Tests')
    parser.add_argument('--batch_sum', type=int, default=16384)
    parser.add_argument('--batch_count', type=int, default=4)
    parser.add_argument('--K', type=int, default=8192)
    parser.add_argument('--N', type=int, default=13824)
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16'])
    args = parser.parse_args()
    dtype = torch.float16

    success_nt = test_grouped_gemm_nt(args.batch_sum, args.batch_count, args.K, args.N, dtype)
    if not success_nt:
        return False
    success_nn = test_grouped_gemm_nn(args.batch_sum, args.batch_count, args.K, args.N, dtype)
    if not success_nn:
        return False
    success_tn = test_grouped_gemm_tn(args.batch_sum, args.batch_count, args.K, args.N, dtype)
    if not success_tn:
        return False
    success_tt = test_grouped_gemm_tt(args.batch_sum, args.batch_count, args.K, args.N, dtype)
    if not success_tt:
        return False
    print("\nAll tests completed successfully!")

    return True


if __name__ == "__main__":
    main()
