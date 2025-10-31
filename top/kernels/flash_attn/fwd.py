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


# Sparse MLA


def _sparse_mla_fwd_kernel(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    kv_stride,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    CP0=True
):
    '''
    This code implements sparse attn
    Note that the first kv_stride - 1 token's out would be nan. since this isn't used, we assume it doesn't matter. (**still, one might have to handle carefully in backward to avoid dout * nan propagated!**)
    It might be OK to set these nan to zero, but we assume it might serve as a reminder of taking care of these out in 'delta = out * dout'.
    The above feature might be replaced with out being undefined if we fix CP0 logic (this logic is currently wrong due to some bug in compiler)
    '''
    assert dim == tilelang.math.next_power_of_2(
        dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(
        tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, 'non-casual is not supported'
    assert topk % block_I == 0, 'otherwise will load some index=0 thus causing wrong kv to be loaded'
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim))**0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    
    @tilelang.jit(
    out_idx=[-1],
    compile_flags=[
        "--use_fast_math",
        "-O3", "-Wno-deprecated-declarations", "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--ptxas-options=-v,--register-usage-level=10", "-DNDEBUG"
        ],
    )
    def _sparse_mla_fwd_func(block_I, threads):

        G = kv_group
        heads = head_kv
        # print(f'heads = {heads}')
        padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
        if padded_H != heads:
            assert kv_group == 1, 'here we solve the heads padding automically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automically)'
        # print(f'padded_H = {padded_H}, heads = {heads}')
        BI = block_I
        NI = tilelang.cdiv(topk, block_I)
        assert NI % 2 == 0, 'NI should be a multiple of 2'
        D = dim
        D_tail = tail_dim
        # Q0 = q_start_index_s
        KV_stride = kv_stride
        # CP0 = q_start_index_s == 0
        # if CP0:
        #     seq_len -= kv_stride - 1
        if head_kv > 64:
            assert head_kv % 64 == 0, 'head_kv should be a multiple of 64'
            REPLICATE_H = head_kv // 64
        else:
            REPLICATE_H = 1

        H_per_block = padded_H if REPLICATE_H == 1 else 64
        @T.prim_func
        def _sparse_mla_fwd_main(
                Q: T.Tensor(q_shape, dtype),  # type: ignore
                KV: T.Tensor(kv_shape, dtype),  # type: ignore
                Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
                q_start_index_s: T.Tensor(1, indices_dtype),
                Output: T.Tensor(o_shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                (seq_len - kv_stride + 1 if CP0 else seq_len) * REPLICATE_H,
                    batch,
                    kv_group,
                    threads=threads) as (bx, by, bz):
                Q_shared_l = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_shared_r = T.alloc_shared([H_per_block, D // 2], dtype)
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
                KV_shared_0_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_0_r = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_l = T.alloc_shared([BI, D // 2], dtype)
                KV_shared_1_r = T.alloc_shared([BI, D // 2], dtype)
                K_tail_shared_0 = T.alloc_shared([BI, D_tail], dtype)
                K_tail_shared_1 = T.alloc_shared([BI, D_tail], dtype)
                O_shared_l = Q_shared_l
                O_shared_r = Q_shared_r
                is_kv_valid = T.alloc_shared([BI], "bool", scope="shared")

                acc_o_l = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_o_r = T.alloc_fragment([H_per_block, D // 2], accum_dtype)
                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sum_exp_shared = T.alloc_shared([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha_shared = T.alloc_shared([H_per_block], accum_dtype, scope="shared")
                alpha_local = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                indices_local = T.alloc_local([1], indices_dtype)

                # TODO: Multi buffer
                bar_q = T.alloc_barrier(arrive_count=384)
                bar_k_0_ready = T.alloc_barrier(arrive_count=128)
                bar_k_1_ready = T.alloc_barrier(arrive_count=128)
                bar_k_0_free = T.alloc_barrier(arrive_count=256)
                bar_k_1_free = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_ready = T.alloc_barrier(arrive_count=256)
                bar_sScale_and_sS_free = T.alloc_barrier(arrive_count=256)

                b_i, g_i = by, bz
                s_i = (bx + (KV_stride - 1 if CP0 else 0)) if REPLICATE_H == 1 else (
                    bx // REPLICATE_H + (KV_stride - 1 if CP0 else 0))
                q_i = q_start_index_s[0] + s_i
                max_kv_i = (q_i + 1 - KV_stride) // KV_stride

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
                H1 = H0 + H_per_block

                tx = T.get_thread_binding()

                T.copy(Q[b_i, s_i, H0:H1, 0:D // 2], Q_shared_l)
                T.copy(Q[b_i, s_i, H0:H1, D // 2:D], Q_shared_r)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)
                T.barrier_arrive(bar_q)

                if tx < 128:
                    T.set_max_nreg(240, 1)
                    T.fill(sumexp, 0)
                    T.fill(m_i, -2**30)  # avoid -inf - inf to cause nan
                    T.fill(acc_o_l, 0)
                    T.barrier_wait(bar_q, 0)

                    for i_i in T.serial(T.ceildiv(NI, 2)):

                        # Buffer 0
                        T.barrier_wait(bar_k_0_ready[0], (i_i & 1))

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0,
                                                            -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_0_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_0_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_0, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        if i_i != 0:
                            T.barrier_arrive(bar_sScale_and_sS_free)
                            T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_0_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_0_free[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_ready[0], (i_i & 1))

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(is_kv_valid[bi_i], 0,
                                                            -T.infinity(acc_s.dtype))
                        T.gemm(Q_shared_l, KV_shared_1_l, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_shared_r, KV_shared_1_r, acc_s, transpose_B=True, wg_wait=-1)
                        T.gemm(Q_tail_shared, K_tail_shared_1, acc_s, transpose_B=True, wg_wait=-1)

                        T.wait_wgmma(0)

                        T.barrier_arrive(bar_sScale_and_sS_free)
                        T.barrier_wait(bar_sScale_and_sS_free, ((i_i * 2 + 1) & 1) ^ 1)

                        T.copy(m_i, m_i_prev)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            alpha_local[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                        T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha_local[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_l[h_i, d_i] *= alpha_local[h_i]
                        T.copy(alpha_local, alpha_shared)

                        T.copy(acc_s, S_shared)
                        T.gemm(S_shared, KV_shared_1_l, acc_o_l)

                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_arrive(bar_k_1_free[0])

                    # Rescale
                    for h_i in T.Parallel(H_per_block):
                        sum_exp_shared[h_i] = sumexp[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_l[h_i, d_i] /= sumexp[h_i]
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(acc_o_l, O_shared_l)
                    T.copy(O_shared_l, Output[b_i, s_i, H0:H1, 0:D // 2])

                elif tx >= 128 and tx < 256:
                    T.set_max_nreg(168, 1)
                    T.fill(acc_o_r, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_0_r, acc_o_r)
                        T.barrier_arrive(bar_k_0_free[0])
                        T.barrier_arrive(bar_sScale_and_sS_free)

                        # Buffer 1
                        T.barrier_arrive(bar_sScale_and_sS_ready)
                        T.barrier_wait(bar_sScale_and_sS_ready, ((i_i * 2 + 1) & 1))
                        for h_i, d_i in T.Parallel(H_per_block, D // 2):
                            acc_o_r[h_i, d_i] *= alpha_shared[h_i]
                        T.gemm(S_shared, KV_shared_1_r, acc_o_r)
                        T.barrier_arrive(bar_k_1_free[0])
                        if i_i != T.ceildiv(NI, 2) - 1:
                            T.barrier_arrive(bar_sScale_and_sS_free)

                    # Rescale
                    for h_i, d_i in T.Parallel(H_per_block, D // 2):
                        acc_o_r[h_i, d_i] /= sum_exp_shared[h_i]

                    T.copy(acc_o_r, O_shared_r)
                    T.copy(O_shared_r, Output[b_i, s_i, H0:H1, D // 2:D])
                elif tx >= 256:
                    # producer
                    T.set_max_nreg(80, 0)
                    for i_i in T.serial(T.ceildiv(NI, 2)):
                        # Buffer 0
                        T.barrier_wait(bar_k_0_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            indices_local[0] = Indices[b_i, s_i, g_i,
                                                    (i_i * 2) * BI + r * 16 + (tx - 256) // 8]
                            is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                            if is_kv_valid[r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_0_l[r * 16 + (tx - 256) // 8,
                                                        64 * u + (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_0_r[r * 16 + (tx - 256) // 8,
                                                        64 * u + (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i, D // 2 +
                                                                64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_0[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_0_ready[0])

                        # Buffer 1
                        T.barrier_wait(bar_k_1_free[0], ((i_i & 1) ^ 1))
                        for r in T.serial(4):
                            indices_local[0] = Indices[b_i, s_i, g_i,
                                                    (i_i * 2 + 1) * BI + r * 16 + (tx - 256) // 8]
                            is_kv_valid[r * 16 + (tx - 256) // 8] = indices_local[0] <= max_kv_i
                            if is_kv_valid[r * 16 + (tx - 256) // 8]:
                                with T.attr("default", "async_scope", 1):
                                    for u in T.serial(4):
                                        for v in T.vectorized(8):
                                            KV_shared_1_l[r * 16 + (tx - 256) // 8,
                                                        64 * u + (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                64 * u + (tx - 256) % 8 * 8 + v]
                                            KV_shared_1_r[r * 16 + (tx - 256) // 8,
                                                        64 * u + (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i, D // 2 +
                                                                64 * u + (tx - 256) % 8 * 8 + v]
                                with T.attr("default", "async_scope", 1):
                                    for v in T.vectorized(8):
                                        K_tail_shared_1[r * 16 + (tx - 256) // 8, (tx - 256) % 8 * 8 +
                                                        v] = KV[b_i, indices_local[0], g_i,
                                                                D + (tx - 256) % 8 * 8 + v]
                        T.cp_async_barrier_noinc(bar_k_1_ready[0])

        return _sparse_mla_fwd_main
      
    return _sparse_mla_fwd_func


@torch.library.custom_op("top::sparse_mla_fwd_wrapped_kernel", mutates_args=())
def _sparse_mla_fwd_wrapped_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    kv_stride: int,
    kv_group: int,
    sm_scale: Optional[float],
    is_causal: bool,
    CP0: bool,
    block_I: int,
    threads: int,
    Q: torch.Tensor,
    KV: torch.Tensor,
    Indices: torch.Tensor,
    q_start_index_s: torch.Tensor,
) -> torch.Tensor:
    return _sparse_mla_fwd_kernel(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride,
                                  kv_group, sm_scale, is_causal, CP0)(block_I, threads)(Q, KV, Indices,
                                                                                  q_start_index_s)


@_sparse_mla_fwd_wrapped_kernel.register_fake
def _(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_stride, kv_group, sm_scale, is_causal, CP0, block_I, threads, *inputs):
    fake_o = torch.empty([batch, seq_len, heads, dim], device=inputs[0].device,
                         dtype=inputs[0].dtype)
    return fake_o


class sparse_mla_fwd_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 seq_len,
                 seq_len_kv,
                 heads,
                 dim,
                 tail_dim,
                 topk,
                 kv_stride,
                 kv_group=1,
                 sm_scale=None,
                 is_causal=True,
                 CP0=True,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.dim = dim
        self.tail_dim = tail_dim
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.CP0 = CP0

        self.kernel = _sparse_mla_fwd_kernel(self.batch, self.seq_len, self.seq_len_kv, self.heads,
                                             self.dim, self.tail_dim, self.topk, self.kv_stride,
                                             self.kv_group, self.sm_scale, self.is_causal, self.CP0)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"block_I": 64, "threads": 384}
    
    @property
    def autotune_configs(self) -> list[dict]:
        block_I = [32, 64, 128]
        threads = [256, 384, 512]
        _configs = list(itertools.product(block_I, threads))

        configs = [{
            'block_I': c[0],
            'threads': c[1],
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, KV: torch.Tensor, Indices: torch.Tensor, q_start_index_s: torch.Tensor):
        return _sparse_mla_fwd_wrapped_kernel(self.batch, self.seq_len, self.seq_len_kv, self.heads, self.dim,
                                             self.tail_dim, self.topk, self.kv_stride, self.kv_group, self.sm_scale,
                                             self.is_causal, self.CP0, self.config["block_I"], self.config["threads"],
                                             Q, KV, Indices, q_start_index_s)
    