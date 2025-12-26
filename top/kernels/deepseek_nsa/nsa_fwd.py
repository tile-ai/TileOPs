import tilelang
import tilelang.language as T
from typing import Optional, Tuple
from top.kernels.kernel import Kernel
import itertools
import torch

from top.kernels.deepseek_nsa.nsa_torch import naive_nsa

__all__ = ["nsa_fwd_kernel"]


# dtype default float16, accum_dtype default float32
def _nsa_fwd_kernel(
    batch, 
    heads, 
    seq_len, 
    dim, 
    is_causal, 
    scale=None, 
    block_size=64, 
    groups=1, 
    selected_blocks=16
    ):
    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32

    block_S = block_size
    # block_T = min(128, tilelang.math.next_power_of_2(dim))
    # num_stages = 2
    # threads = 32
    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    
    def _nsa_fwd_func(block_T, num_stages, threads):

        NK = tilelang.cdiv(dim, block_T)
        NV = tilelang.cdiv(dim, block_T)
        assert NK == 1, "The key dimension can not be larger than 256"

        S = selected_blocks
        G = groups
        BS = block_S
        BK = BV = block_T

        q_shape = [batch, seq_len, heads, dim]
        kv_shape = [batch, seq_len, head_kv, dim]
        block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
        
        @T.prim_func
        def _nsa_fwd_main(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([G, BK], dtype)
                K_shared = T.alloc_shared([BS, BK], dtype)
                V_shared = T.alloc_shared([BS, BV], dtype)
                O_shared = T.alloc_shared([G, BV], dtype)

                acc_s = T.alloc_fragment([G, BS], accum_dtype)
                acc_s_cast = T.alloc_fragment([G, BS], dtype)
                acc_o = T.alloc_fragment([G, BV], accum_dtype)
                scores_max = T.alloc_fragment([G], accum_dtype)
                scores_max_prev = T.alloc_fragment([G], accum_dtype)
                scores_scale = T.alloc_fragment([G], accum_dtype)
                scores_sum = T.alloc_fragment([G], accum_dtype)
                logsum = T.alloc_fragment([G], accum_dtype)

                i_t, i_v, i_bh = bx, by, bz
                i_b, i_h = i_bh // head_kv, i_bh % head_kv

                NS = S
                T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for i in T.Pipelined(NS, num_stages=num_stages):
                    i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                    if i_s <= i_t and i_s >= 0:
                        # [BS, BK]
                        T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                        if is_causal:
                            for i, j in T.Parallel(G, BS):
                                acc_s[i, j] = T.if_then_else(i_t >= (i_s + j), 0, -T.infinity(acc_s.dtype))
                        else:
                            T.clear(acc_s)

                        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                        # Softmax
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(accum_dtype))
                        T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                        for i in T.Parallel(G):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                        for i, j in T.Parallel(G, BS):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(G):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        T.copy(acc_s, acc_s_cast)

                        # Rescale
                        for i, j in T.Parallel(G, BV):
                            acc_o[i, j] *= scores_scale[i]

                        # V * softmax(Q * K)
                        T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(G, BV):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[i_b, i_t, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

        return _nsa_fwd_main
    return _nsa_fwd_func


@torch.library.custom_op("top::nsa_fwd_wrapped_kernel", mutates_args=())
def _nsa_fwd_wrapped_kernel(
    batch: int,
    heads: int,
    seq_len: int,
    dim: int,
    is_causal: bool,
    scale: float,
    block_size: int,
    groups: int,
    selected_blocks: int,
    block_T: int,
    num_stages: int,
    threads: int,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    BlockIndices: torch.Tensor,
) ->torch.Tensor:
    return _nsa_fwd_kernel(batch, heads, seq_len, dim, is_causal, scale, block_size, 
                            groups, selected_blocks)(block_T, num_stages, 
                            threads)(Q, K, V, BlockIndices)


@_nsa_fwd_wrapped_kernel.register_fake
def _(
    batch, 
    heads, 
    seq_len, 
    dim, 
    is_causal, 
    scale, 
    block_size, 
    groups, 
    selected_blocks, 
    block_T, 
    num_stages, 
    threads, 
    *inputs
) -> torch.Tensor:
    fake_o = torch.empty_like(inputs[0])
    return fake_o


class nsa_fwd_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90, 100]

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
        config: Optional[dict] = None,
        tune=False):

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

        self.kernel = _nsa_fwd_kernel(self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.scale, self.block_size, self.groups, self.selected_blocks)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_T": min(128, tilelang.math.next_power_of_2(self.dim)),
            "num_stages": 2,
            "threads": 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_T = [32, 64, 128]
        num_stages = [2,]
        threads = [32, 64]
        _configs = list(itertools.product(block_T, num_stages, threads))
        configs = [{
            "block_T": c[0],
            "num_stages": c[1],
            "threads": c[2]
        } for c in _configs]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, BlockIndices: torch.Tensor):
        return _nsa_fwd_wrapped_kernel(self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.scale, self.block_size, self.groups, self.selected_blocks, self.config["block_T"], self.config["num_stages"], self.config["threads"], Q, K, V, BlockIndices)


def main():
    B, SEQ_LEN, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1

    block_T = min(128, tilelang.math.next_power_of_2(D))
    kernel = _nsa_fwd_kernel(
        batch=B,
        heads=HQ,
        seq_len=SEQ_LEN,
        dim=D,
        is_causal=True,
        scale=scale,
        block_size=block_size,
        groups=HQ // H,
        selected_blocks=S,
    )(block_T=block_T, num_stages=2, threads=32)

    kernel2 = nsa_fwd_kernel(
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


    src_kernel = kernel.get_kernel_source()
    print(src_kernel)
    # with open("nsa_fwd_kernel.cu", "w") as f:
    #     f.write(src_kernel)
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

    out = kernel(Q, K, V, block_indices.to(torch.int32))

    out2 = kernel2.forward(Q, K, V, block_indices.to(torch.int32))

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
    print("out2", out2)
    print("ref", ref)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ref, out2, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()
