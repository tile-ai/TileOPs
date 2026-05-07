import functools
import torch
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout
from typing import Optional


def _make_moe_grouped_gemm_persistent_tma(
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: str,
    sm_count: int,
):
    _num_pid_n = T.ceildiv(block_n, block_n)  # always 1 for simplicity
    _scan_rounds = (num_experts + 31) // 32

    @T.prim_func
    def tma_kernel(
        A: T.Buffer(None, dtype),
        B: T.Buffer((num_experts, block_n, "K"), dtype),
        true_sizes: T.Buffer((num_experts,), "int32"),
        true_offsets: T.Buffer((num_experts,), "int32"),
        C: T.Buffer(None, dtype),
        tile_counter: T.Buffer((1,), "int32"),
        numel: T.int32,
        output_numel: T.int32,
    ):
        with T.Kernel(sm_count, threads=128) as bx:
            s_cum = T.alloc_shared([num_experts + 1], "int32")
            A_shared = T.alloc_shared([block_m, block_k], dtype)
            B_shared = T.alloc_shared([block_n, block_k], dtype)
            C_acc = T.alloc_fragment([block_m, block_n], "float32")

            tx = T.get_thread_binding(0)
            lane_idx = tx % T.int32(32)

            s_cum[0] = T.int32(0)
            if tx < T.int32(32):
                carry = T.alloc_local([1], "int32")
                carry[0] = T.int32(0)
                for r in T.serial(_scan_rounds):
                    e = r * T.int32(32) + lane_idx
                    val = T.alloc_local([1], "int32")
                    if e < T.int32(num_experts):
                        val[0] = (true_sizes[e] + T.int32(block_m - 1)) // T.int32(block_m)
                    else:
                        val[0] = T.int32(0)
                    for shift in T.serial(5):
                        s = T.int32(1) << shift
                        v = T.shfl_up(val[0], s)
                        if lane_idx >= s:
                            val[0] = val[0] + v
                    smem_idx = r * T.int32(32) + lane_idx + 1
                    if smem_idx <= T.int32(num_experts):
                        s_cum[smem_idx] = val[0] + carry[0]
                    carry[0] = T.call_extern(
                        "int32",
                        "__shfl_sync",
                        0xFFFFFFFF,
                        val[0],
                        T.int32(31),
                    ) + carry[0]
            T.sync_threads()

            total_tiles = s_cum[num_experts]

            for _iter in T.serial(
                (numel // block_m + num_experts) * _num_pid_n // sm_count + 2
            ):
                flat_id = T.alloc_local([1], "int32")
                flat_id[0] = T.call_extern(
                    "int32",
                    "atomicAdd",
                    T.address_of(tile_counter[0]),
                    T.int32(1),
                )
                if flat_id[0] < total_tiles:
                    pid_m = T.alloc_local([1], "int32")
                    pid_n = T.alloc_local([1], "int32")
                    lo = T.alloc_local([1], "int32")
                    lo[0] = T.int32(0)
                    hi = T.alloc_local([1], "int32")
                    hi[0] = T.int32(num_experts)
                    for _bs in T.serial(8):
                        mid = T.alloc_local([1], "int32")
                        mid[0] = (lo[0] + hi[0]) >> T.int32(1)
                        if s_cum[mid[0] + 1] <= flat_id[0]:
                            lo[0] = mid[0] + T.int32(1)
                        else:
                            hi[0] = mid[0]
                    pid_m[0] = flat_id[0] - s_cum[lo[0]]
                    pid_n[0] = T.int32(0)
                    exp_id = lo[0]
                    row_start = true_offsets[exp_id] + pid_m[0] * T.int32(block_m)
                    col_start = pid_n[0] * T.int32(block_n)
                    K = A.shape[1]
                    T.clear(C_acc)
                    for k in T.serial(T.ceildiv(K, block_k)):
                        T.copy(A[row_start : row_start + block_m, k * block_k : (k + 1) * block_k], A_shared)
                        T.copy(B[exp_id, col_start : col_start + block_n, k * block_k : (k + 1) * block_k], B_shared)
                        T.gemm(A_shared, B_shared, C_acc, transpose_B=True)
                    T.copy(C_acc, C[row_start : row_start + block_m, col_start : col_start + block_n])

    return tma_kernel


def _make_moe_grouped_gemm_persistent_notma(
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: str,
    sm_count: int,
):
    _num_pid_n = T.ceildiv(block_n, block_n)  # always 1 for simplicity
    _scan_rounds = (num_experts + 31) // 32

    @T.prim_func
    def notma_kernel(
        A: T.Buffer(None, dtype),
        B: T.Buffer((num_experts, block_n, "K"), dtype),
        true_sizes: T.Buffer((num_experts,), "int32"),
        true_offsets: T.Buffer((num_experts,), "int32"),
        C: T.Buffer(None, dtype),
        tile_counter: T.Buffer((1,), "int32"),
        numel: T.int32,
        output_numel: T.int32,
    ):
        with T.Kernel(sm_count, threads=128) as bx:
            s_cum = T.alloc_shared([num_experts + 1], "int32")
            A_shared = T.alloc_shared([block_m, block_k], dtype)
            B_shared = T.alloc_shared([block_n, block_k], dtype)
            C_acc = T.alloc_fragment([block_m, block_n], "float32")

            tx = T.get_thread_binding(0)
            lane_idx = tx % T.int32(32)

            s_cum[0] = T.int32(0)
            if tx < T.int32(32):
                carry = T.alloc_local([1], "int32")
                carry[0] = T.int32(0)
                for r in T.serial(_scan_rounds):
                    e = r * T.int32(32) + lane_idx
                    val = T.alloc_local([1], "int32")
                    if e < T.int32(num_experts):
                        val[0] = (true_sizes[e] + T.int32(block_m - 1)) // T.int32(block_m)
                    else:
                        val[0] = T.int32(0)
                    for shift in T.serial(5):
                        s = T.int32(1) << shift
                        v = T.shfl_up(val[0], s)
                        if lane_idx >= s:
                            val[0] = val[0] + v
                    smem_idx = r * T.int32(32) + lane_idx + 1
                    if smem_idx <= T.int32(num_experts):
                        s_cum[smem_idx] = val[0] + carry[0]
                    carry[0] = T.call_extern(
                        "int32",
                        "__shfl_sync",
                        0xFFFFFFFF,
                        val[0],
                        T.int32(31),
                    ) + carry[0]
            T.sync_threads()

            total_tiles = s_cum[num_experts]

            for _iter in T.serial(
                (numel // block_m + num_experts) * _num_pid_n // sm_count + 2
            ):
                flat_id = T.alloc_local([1], "int32")
                flat_id[0] = T.call_extern(
                    "int32",
                    "atomicAdd",
                    T.address_of(tile_counter[0]),
                    T.int32(1),
                )
                if flat_id[0] < total_tiles:
                    pid_m = T.alloc_local([1], "int32")
                    pid_n = T.alloc_local([1], "int32")
                    lo = T.alloc_local([1], "int32")
                    lo[0] = T.int32(0)
                    hi = T.alloc_local([1], "int32")
                    hi[0] = T.int32(num_experts)
                    for _bs in T.serial(8):
                        mid = T.alloc_local([1], "int32")
                        mid[0] = (lo[0] + hi[0]) >> T.int32(1)
                        if s_cum[mid[0] + 1] <= flat_id[0]:
                            lo[0] = mid[0] + T.int32(1)
                        else:
                            hi[0] = mid[0]
                    pid_m[0] = flat_id[0] - s_cum[lo[0]]
                    pid_n[0] = T.int32(0)
                    exp_id = lo[0]
                    row_start = true_offsets[exp_id] + pid_m[0] * T.int32(block_m)
                    col_start = pid_n[0] * T.int32(block_n)
                    K = A.shape[1]
                    T.clear(C_acc)
                    for k in T.serial(T.ceildiv(K, block_k)):
                        T.copy(A[row_start : row_start + block_m, k * block_k : (k + 1) * block_k], A_shared)
                        T.copy(B[exp_id, col_start : col_start + block_n, k * block_k : (k + 1) * block_k], B_shared)
                        T.gemm(A_shared, B_shared, C_acc, transpose_B=True)
                    T.copy(C_acc, C[row_start : row_start + block_m, col_start : col_start + block_n])

    return notma_kernel


class MoeGroupedGemmPersistentKernel:
    def __init__(
        self,
        T: int,
        E: int,
        N: int,
        K: int,
        top_k: int,
        dtype: str = "float16",
        block_m: int = 64,
        block_n: int = 128,
        block_k: int = 64,
        num_stages: int = 2,
        sm_count: int = 132,
    ):
        self.T = T
        self.E = E
        self.N = N
        self.K = K
        self.top_k = top_k
        self.dtype = dtype
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k if K % block_k == 0 else K
        self.num_stages = num_stages
        self.sm_count = sm_count
        self._tile_counter = torch.zeros(1, dtype=torch.int32, device="cuda")

    @functools.cached_property
    def _tma_kernel(self):
        return _make_moe_grouped_gemm_persistent_tma(
            num_experts=self.E,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_stages=self.num_stages,
            dtype=self.dtype,
            sm_count=self.sm_count,
        )

    @functools.cached_property
    def _notma_kernel(self):
        return _make_moe_grouped_gemm_persistent_notma(
            num_experts=self.E,
            block_m=self.block_m,
            block_n=self.block_n,
            block_k=self.block_k,
            num_stages=self.num_stages,
            dtype=self.dtype,
            sm_count=self.sm_count,
        )

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        true_sizes: torch.Tensor,
        true_offsets: torch.Tensor,
    ) -> torch.Tensor:
        numel = A.shape[0]
        output_numel = numel
        C = torch.zeros(
            (output_numel, self.N), dtype=A.dtype, device=A.device
        )
        self._tile_counter.fill_(0)

        use_tma = (self.K % self.block_k == 0)

        if use_tma:
            kernel = self._tma_kernel
            A_padded = torch.zeros(
                (numel + self.block_m, self.K), dtype=A.dtype, device=A.device
            )
            A_padded[:numel] = A
            kernel(
                A_padded,
                B,
                true_sizes,
                true_offsets,
                C,
                self._tile_counter,
                numel,
                output_numel,
            )
        else:
            kernel = self._notma_kernel
            kernel(
                A,
                B,
                true_sizes,
                true_offsets,
                C,
                self._tile_counter,
                numel,
                output_numel,
            )

        return C

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
