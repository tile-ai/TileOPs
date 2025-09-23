import torch
import torch.backends
import tilelang
import tilelang.language as T
import torch.nn as nn
from tilelang import tvm as tvm
from tvm import DataType
from tilelang.intrinsics.mma_layout import (
    make_mma_swizzle_layout as make_swizzle_layout,)
from tilelang.language.utils import index_to_coordinates
import numpy as np

from tilelang.intrinsics.mma_macro_generator import (
    INT4TensorCoreIntrinEmitter,)
from tilelang.transform import simplify_prim_func

decode_i2s_to_i8s = """template <typename T1, typename T2>
__device__ void decode_i2s_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024
    static constexpr uint MEDIAN_NUM = 0x02020202;
#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
        i8s[i] = __vsub4(i8s[i], MEDIAN_NUM);
    }
}
template <typename T1, typename T2>
__device__ void decode_i2u_to_i8s(T1 *_i2b, T2 *_i8s, const int N = 16)
{
    // convert 8 int2b_t to 8 int8b_t -> 2 int32
    uint *i8s = reinterpret_cast<uint *>(_i8s);

    // i2b = {e7,e6,e5,e4,e3,e2,e1,e0}
    // also require interleave {e7,e3,e6,e2,e5,e1,e4,e0}
    uint const i2b = *reinterpret_cast<uint *>(_i2b);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint immLut = (0xf0 & 0xcc) | 0xaa; // 0b11101010
    static constexpr uint BOTTOM_MASK = 0x03030303;      // 0xf -> 0b11 select 0,3
    static constexpr uint I8s_MAGIC_NUM = 0x00000000;    // 1024

#pragma unroll
    for (int i = 0; i < (N / 4); i++)
    {
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\\n"
                     : "=r"(i8s[i])
                     : "r"(i2b >> (2 * i)), "n"(BOTTOM_MASK), "n"(I8s_MAGIC_NUM), "n"(immLut));
    }
}
"""


@simplify_prim_func
def _bitnet_158_int8xint2_prefill(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    fast_decoding=True,
    block_row_warps=2,
    block_col_warps=2,
    warp_row_tiles=32,
    warp_col_tiles=32,
    chunk=64,
):
    assert in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        "float16",
        "float32",
        "int32",
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if accum_dtype == "int32":
        micro_size_k = 32

    num_elems_per_byte = 4
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    local_size_compressed = local_size // num_elems_per_byte

    shared_scope = "shared.dyn"
    storage_dtype = "int8"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)  # int8 storage represents int4*2
    B_shape = (N, K // num_elems_per_byte)  # int8 storage represents int4*2
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequantize_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    fragement_size_a = (micro_size_x * micro_size_k) // warp_size
    fragement_size_b = (micro_size_y * micro_size_k) // warp_size
    fragement_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = INT4TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    @T.prim_func
    def _bitnet_158_int8xint2_prefill_main(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, block_N),
                T.ceildiv(M, block_M),
                threads=threads,
                prelude=decode_i2s_to_i8s,
        ) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
            B_dequantize_shared = T.alloc_shared(
                B_dequantize_shared_shape, in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
            A_frag = T.alloc_local((warp_rows * fragement_size_a), in_dtype)
            B_frag = T.alloc_local((warp_cols * fragement_size_b), in_dtype)
            C_frag = T.alloc_local((warp_rows * warp_cols * fragement_size_c), accum_dtype)

            B_local = T.alloc_local([local_size_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([local_size], in_dtype)

            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_dequantize_shared: make_swizzle_layout(B_dequantize_shared),
            })

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_frag)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K // num_elems_per_byte):
                    B_shared[j, k] = B[bx * block_N + j, ko * (block_K // num_elems_per_byte) + k]

                for i in T.serial(block_N * block_K // num_elems_per_byte //
                                  (threads * local_size_compressed)):
                    for v in T.vectorized(0, local_size_compressed):
                        index = (
                            i * threads * local_size_compressed +
                            thread_bindings * local_size_compressed + v)
                        vi, vj = index_to_coordinates(index, B_shared_shape)
                        B_local[v] = B_shared[vi, vj]

                    T.call_extern(
                        "handle",
                        "decode_i2u_to_i8s",
                        T.address_of(B_local[0]),
                        T.address_of(B_dequantize_local[0]),
                    )

                    for v in T.vectorized(0, local_size):
                        index = (i * threads * local_size + thread_bindings * local_size + v)
                        vi, vj = index_to_coordinates(index, B_dequantize_shared_shape)
                        B_dequantize_shared[vi, vj] = B_dequantize_local[v]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_frag,
                        A_shared,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_frag,
                        B_dequantize_shared,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_frag, B_frag, C_frag)

            # Perform STMatrix
            mma_emitter.stmatrix(
                C_frag,
                C_shared,
            )

            # Store shared into global
            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x,
                    j // micro_size_y,
                    i % micro_size_x,
                    j % micro_size_y,
                ]

    return _bitnet_158_int8xint2_prefill_main


@simplify_prim_func
def _bitnet_158_int8xint2_decode(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    fast_decoding=True,
    n_partition=4,
    reduce_thread=32,
):
    assert in_dtype in [
        "float16",
        "int8",
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        "float16",
        "float32",
        "int32",
    ], "Currently only float16, float32 and int32 are supported"
    storage_nbit = 8
    num_bits = 2
    A_shape = (M, K)
    B_shape = (N, K // storage_nbit * num_bits)
    C_shape = (M, N)

    num_elems_per_byte = 4
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    storage_dtype = "int8"
    block_K = reduce_thread * micro_size_k

    use_dp4a = True
    dp4a_size = 4

    @T.prim_func
    def _bitnet_158_int8xint2_decode_kernel(
            A: T.Buffer(A_shape, in_dtype),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer(C_shape, out_dtype),
    ):
        with T.Kernel(
                T.ceildiv(N, n_partition),
                M,
                threads=(reduce_thread, n_partition),
        ) as (
                bx,
                by,
        ):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(decode_i2s_to_i8s)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed) +
                        kr * micro_size_k_compressed + v,
                    ]

                T.call_extern(
                    "handle",
                    "decode_i2u_to_i8s",
                    T.address_of(B_quant_local[0]),
                    T.address_of(B_dequantize_local[0]),
                )

                if use_dp4a:
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_dequantize_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki] * B_dequantize_local[ki]

            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    ))
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]

    return _bitnet_158_int8xint2_decode_kernel


class Bitnet_158_int8xint2_kernel(nn.Module):

    def __init__(self,
                 M,
                 N,
                 K,
                 in_dtype,
                 out_dtype,
                 accum_dtype,
                 fast_decoding=True):
        super().__init__()
        self.in_dtype = in_dtype
        self.accum_dtype = accum_dtype
        self.prefill_program = _bitnet_158_int8xint2_prefill(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
        self.prefill_kernel = tilelang.compile(self.prefill_program)
        
        self.decode_program = _bitnet_158_int8xint2_decode(M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
        self.decode_kernel = tilelang.compile(self.decode_program)
        
    @staticmethod
    def general_compress(lowprecision_weight, source_bits=4, storage_dtype=np.int8):
        elems_per_byte = 8 // source_bits
        if lowprecision_weight.dtype == np.float16:
            lowprecision_weight = lowprecision_weight.astype(dtype=np.int8)
        int8_weight = np.zeros(
            (
                *lowprecision_weight.shape[:-1],
                lowprecision_weight.shape[-1] // elems_per_byte,
            ),
            dtype=np.int8,
        )
        for j in range(lowprecision_weight.shape[-1] // elems_per_byte):
            for k in range(elems_per_byte):
                int8_weight[:, j] |= lowprecision_weight[:, j * elems_per_byte + k] << (source_bits * k)

        return int8_weight.view(storage_dtype)


    # interleave weight numpy implementation
    @staticmethod
    def interleave_weight(qweight, nbits=4, target_dtype="float16"):
        assert target_dtype in ["float16", "int8"]
        # reinterpret the data type of qweight to int32
        qweight = qweight.view(np.int32)
        new_qweight = np.zeros_like(qweight)
        bits_stride = 8 if target_dtype == "int8" else 16
        mask = (1 << nbits) - 1  # for 4bit the val is 0x0000000f
        num_groups = 32 // bits_stride
        elems_per_group = bits_stride // nbits
        for i in range(num_groups):
            for j in range(elems_per_group):
                offset = i * elems_per_group + j
                shift = (offset % num_groups) * bits_stride + (offset // num_groups) * nbits
                new_qweight |= ((qweight >> (nbits * offset)) & mask) << shift

        if nbits == 1 and target_dtype == "int8":
            # special handling for 1b interleave
            n16_weight = new_qweight & np.int32(0xF0F00F0F)
            n16_weight |= ((new_qweight & np.int32(0x000000F0)) >> 4) << 16
            n16_weight |= ((new_qweight & np.int32(0x0000F000)) >> 12) << 24
            n16_weight |= ((new_qweight & np.int32(0x000F0000)) >> 16) << 4
            n16_weight |= ((new_qweight & np.int32(0x0F000000)) >> 24) << 12
            return n16_weight.view(np.int8)
        elif nbits == 2 and target_dtype == "float16":
            n8_weight = new_qweight & np.int32(0xFF0000FF)
            n8_weight |= ((new_qweight & np.int32(0x0000FF00)) >> 8) << 16
            n8_weight |= ((new_qweight & np.int32(0x00FF0000)) >> 16) << 8
            return n8_weight.view(np.int8)
        elif nbits == 1 and target_dtype == "float16":
            n8_weight = new_qweight & 0xF000000F
            n8_weight |= ((new_qweight & 0x000000F0) >> 4) << 8
            n8_weight |= ((new_qweight & 0x00000F00) >> 8) << 16
            n8_weight |= ((new_qweight & 0x0000F000) >> 12) << 24
            n8_weight |= ((new_qweight & 0x000F0000) >> 16) << 4
            n8_weight |= ((new_qweight & 0x00F00000) >> 20) << 12
            n8_weight |= ((new_qweight & 0x0F000000) >> 24) << 20

        return new_qweight.view(np.int8)
    
    def get_B_int8(self, B):
        qw = self.general_compress(B.cpu().numpy(), source_bits=2, storage_dtype=np.int8)
        qw = self.interleave_weight(qw, 2, target_dtype=self.in_dtype)
        qw = torch.from_numpy(qw).to(device="cuda")
        return qw

    def prefill(self, A, B, C):
        src_code = self.prefill_kernel.get_kernel_source()
        # src_code is the generated cuda source
        assert src_code is not None
        # print(src_code)
        qw = self.get_B_int8(B)
        self.prefill_kernel(A, qw, C)
        return C

    def decode(self, A, B, C):
        src_code = self.decode_kernel.get_kernel_source()
        # src_code is the generated cuda source
        assert src_code is not None
        # print(src_code)
        qw = self.get_B_int8(B)
        self.decode_kernel(A, qw, C)
        return C

    def profile(self, warmup=500):
        pass
        # latency = self.bwd_profiler.do_bench(warmup=warmup)
        # return latency

    def ref_program(self, A, B):
        ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, self.accum_dtype))
        return ref_c

    def check(self, A, B, C, mode):
        if mode == "prefill":
            C = self.prefill(A, B, C)
        elif mode == "decode":
            C = self.decode(A, B, C)
        else:
            raise ValueError("mode should be either 'prefill' or 'decode'")
        
        ref_c = self.ref_program(A, B)
        torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
        print(f"Bitnet_158_int8xint2 {mode} kernel check passed!")
        