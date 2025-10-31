import tilelang
import tilelang.language as T
from typing import Optional, Tuple
from top.kernels.kernel import Kernel
import itertools
import torch

__all__ = [
    'mha_fwd_kernel', 'mha_fwd_wgmma_pipelined_kernel', 'gqa_fwd_kernel',
    'gqa_fwd_wgmma_pipelined_kernel'
]

# MHA


def _mha_fwd_kernel(batch, heads, seq_len, dim, is_causal, dtype='float16'):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_fwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_fwd_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                Output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _mha_fwd_main

    return _mha_fwd_func


@torch.library.custom_op("top::mha_fwd_wrapped_kernel", mutates_args=())
def _mha_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _mha_fwd_kernel(batch, heads, seq_len, dim, is_causal,
                           dtype)(block_M, block_N, num_stages, threads)(Q, K, V)


@_mha_fwd_wrapped_kernel.register_fake
def _(batch, heads, seq_len, dim, is_causal, dtype, block_M, block_N, num_stages, threads, *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class mha_fwd_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_fwd_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return _mha_fwd_wrapped_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                       self.is_causal, self.dtype_str, self.config["block_M"],
                                       self.config["block_N"], self.config["num_stages"],
                                       self.config["threads"], Q, K, V)


def _mha_fwd_wgmma_pipelined_kernel(batch, heads, seq_len, dim, is_causal, dtype="float16"):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_fwd_wgmma_pipelined_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def _mha_fwd_wgmma_pipelined_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                Output: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({O_shared: tilelang.layout.make_swizzled_layout(O_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _mha_fwd_wgmma_pipelined_main

    return _mha_fwd_wgmma_pipelined_func


@torch.library.custom_op("top::mha_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _mha_fwd_wgmma_pipelined_wrapped_kernel(
    batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
    dtype: str, block_M: int, block_N: int, num_stages: int, threads: int,
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _mha_fwd_wgmma_pipelined_kernel(batch, heads, seq_len, dim, is_causal, dtype)(
        block_M, block_N, num_stages, threads)(Q, K, V)


@_mha_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch, heads, seq_len, dim, is_causal, dtype, block_M, block_N, num_stages, threads, *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class mha_fwd_wgmma_pipelined_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_fwd_wgmma_pipelined_kernel(self.batch, self.heads, self.seq_len,
                                                      self.dim, self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return _mha_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                       self.is_causal, self.dtype_str, self.config["block_M"],
                                       self.config["block_N"], self.config["num_stages"],
                                       self.config["threads"], Q, K, V)


# GQA


def _gqa_fwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype='float16'):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
    groups = heads // heads_kv
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads_kv, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _gqa_fwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(kv_shape, dtype),  # type: ignore
                V: T.Tensor(kv_shape, dtype),  # type: ignore
                Output: T.Tensor(q_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                         -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _gqa_fwd_main

    return _gqa_fwd_func


@torch.library.custom_op("top::gqa_fwd_wrapped_kernel", mutates_args=())
def _gqa_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_kernel(batch, heads, heads_kv, seq_len, dim, is_causal,
                          dtype)(block_M, block_N, num_stages, threads)(Q, K, V)


@_gqa_fwd_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, seq_len, dim, is_causal, dtype, block_M, block_N, num_stages, threads, *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class gqa_fwd_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 heads_kv,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_kernel(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 64 if self.dim <= 128 else 32,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return _gqa_fwd_wrapped_kernel(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                                   self.is_causal, self.dtype_str, self.config["block_M"],
                                   self.config["block_N"], self.config["num_stages"],
                                   self.config["threads"], Q, K, V)


def _gqa_fwd_wgmma_pipelined_kernel(batch,
                                    heads,
                                    heads_kv,
                                    seq_len,
                                    dim,
                                    is_causal,
                                    dtype="float16"):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
    groups = heads // heads_kv
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, heads_kv, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[3, 4],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_fwd_wgmma_pipelined_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(kv_shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by // groups, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(kv_shape, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by // groups, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def _gqa_fwd_wgmma_pipelined_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                K: T.Tensor(kv_shape, dtype),  # type: ignore
                V: T.Tensor(kv_shape, dtype),  # type: ignore
                Output: T.Tensor(q_shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({O_shared: tilelang.layout.make_swizzled_layout(O_shared)})
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.ceildiv(
                        (bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M:(bx + 1) * block_M])

        return _gqa_fwd_wgmma_pipelined_main

    return _gqa_fwd_wgmma_pipelined_func


@torch.library.custom_op("top::gqa_fwd_wgmma_pipelined_wrapped_kernel", mutates_args=())
def _gqa_fwd_wgmma_pipelined_wrapped_kernel(
    batch: int,
    heads: int,
    heads_kv: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    dtype: str,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _gqa_fwd_wgmma_pipelined_kernel(batch, heads, heads_kv, seq_len, dim, is_causal, dtype)(
        block_M, block_N, num_stages, threads)(Q, K, V)


@_gqa_fwd_wgmma_pipelined_wrapped_kernel.register_fake
def _(batch, heads, heads_kv, seq_len, dim, is_causal, dtype, block_M, block_N, num_stages, threads, *inputs):
    fake_o = torch.empty_like(inputs[0])
    fake_lse = fake_o.new_empty([batch, heads, seq_len])
    return fake_o, fake_lse


class gqa_fwd_wgmma_pipelined_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 heads,
                 heads_kv,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _gqa_fwd_wgmma_pipelined_kernel(self.batch, self.heads, self.heads_kv,
                                                      self.seq_len, self.dim, self.is_causal,
                                                      self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_M": 128, "block_N": 128, "num_stages": 2, "threads": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        return _gqa_fwd_wgmma_pipelined_wrapped_kernel(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                                                   self.is_causal, self.dtype_str, self.config["block_M"],
                                                   self.config["block_N"], self.config["num_stages"],
                                                   self.config["threads"], Q, K, V)


