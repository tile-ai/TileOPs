"""
Gated DeltaNet decode (single-step recurrence).

Given the previous hidden state S_{t-1} and a single token's (q, k, v, g, beta),
compute the output o_t and the updated state S_t:

    S_t = alpha * S_{t-1} * (I - beta * k k^T) + beta * v k^T
        = alpha * S_{t-1} - alpha * beta * (S_{t-1} @ k) k^T + beta * v k^T
    o_t = S_t @ q

where alpha = exp(g).

Equivalently (matching the chunked forward when chunk_size=1):
    old_val  = S @ k                     # matvec
    v_new    = beta * v - alpha * beta * old_val
    o_inter  = alpha * (S @ q)           # matvec
    o_intra  = (q . k) * v_new           # dot product + scale
    o        = o_inter + o_intra
    S_new    = alpha * S + outer(k, v_new)
"""
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel

__all__ = ["GatedDeltaNetDecodeKernel"]

_LOG2E = 1.4426950408889634


def _gated_deltanet_decode_tl(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    dtype: str = "float32",
):
    """TileLang kernel for single-step Gated DeltaNet decode.

    (q, k, v, g, beta, state) -> (o, new_state)
    """
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-2, -1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _decode_func(num_stages, threads=128):
        @T.macro
        def _decode_body(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                q_s = T.alloc_shared([dim_k], accum_dtype)
                k_s = T.alloc_shared([dim_k], accum_dtype)
                v_s = T.alloc_shared([dim_v], accum_dtype)
                h_s = T.alloc_shared([dim_k, dim_v], accum_dtype)
                old_val = T.alloc_shared([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                o_s = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_shared([1], accum_dtype)

                # Load inputs
                for i in T.Parallel(dim_k):
                    q_s[i] = q[bid, hid, i]
                for i in T.Parallel(dim_k):
                    k_s[i] = k[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_s[i] = v[bid, hid, i]
                for i, j in T.Parallel(dim_k, dim_v):
                    h_s[i, j] = state[bid, hid, i, j]

                g_val = g[bid, hid]
                beta_val = beta[bid, hid]
                alpha = T.exp2(g_val * _LOG2E)  # exp(g)
                alpha_beta = alpha * beta_val

                # q . k dot product (scalar reduction, all threads redundantly)
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_s[kk] * k_s[kk]

                # Fused Steps 1+3a: single pass over h_s computes
                #   old_val = S @ k, o_s = S @ q
                for j in T.Parallel(dim_v):
                    old_val[j] = T.float32(0.0)
                    o_s[j] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    for j in T.Parallel(dim_v):
                        old_val[j] += h_s[kk, j] * k_s[kk]
                        o_s[j] += h_s[kk, j] * q_s[kk]

                # Step 2: v_new = beta * v - alpha * beta * old_val
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * v_s[j] - alpha_beta * old_val[j]

                # Step 3: o = alpha * o_inter + qk_dot * v_new
                for j in T.Parallel(dim_v):
                    o_s[j] = alpha * o_s[j] + qk_dot[0] * v_new[j]

                # Write output
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = o_s[j]

                # Step 4: new_state = alpha * S + outer(k, v_new)
                for i, j in T.Parallel(dim_k, dim_v):
                    new_state[bid, hid, i, j] = alpha * h_s[i, j] + k_s[i] * v_new[j]

        @T.prim_func
        def gated_deltanet_decode(
            q: T.Tensor([batch, head, dim_k], dtype),
            k: T.Tensor([batch, head, dim_k], dtype),
            v: T.Tensor([batch, head, dim_v], dtype),
            g: T.Tensor([batch, head], dtype),
            beta: T.Tensor([batch, head], dtype),
            state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, head, dim_v], dtype),
            new_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            _decode_body(q, k, v, g, beta, state, o, new_state)

        return gated_deltanet_decode

    return _decode_func


@torch.library.custom_op("tileops::gated_deltanet_decode_kernel", mutates_args=())
def _gated_deltanet_decode_wrapped_kernel(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _gated_deltanet_decode_tl(batch, head, dim_k, dim_v, dtype)(
        num_stages, threads
    )
    return kernel_fn(q, k, v, g, beta, state)


@_gated_deltanet_decode_wrapped_kernel.register_fake
def _gated_deltanet_decode_wrapped_kernel_fake(
    batch: int,
    head: int,
    dim_k: int,
    dim_v: int,
    dtype: str,
    num_stages: int,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(batch, head, dim_v, dtype=q.dtype, device=q.device)
    new_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, new_state


class GatedDeltaNetDecodeKernel(Kernel):
    """Gated DeltaNet single-step decode kernel.

    Computes one step of the gated delta rule recurrence:
        S_t = alpha_t * S_{t-1} * (I - beta_t * k_t k_t^T) + beta_t * v_t k_t^T
        o_t = S_t @ q_t
    """

    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch: int,
        head: int,
        dim_k: int,
        dim_v: int,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {"num_stages": 2, "threads": 128}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _gated_deltanet_decode_wrapped_kernel(
            self.batch,
            self.head,
            self.dim_k,
            self.dim_v,
            self.dtype_str,
            self.config["num_stages"],
            self.config["threads"],
            q, k, v, g, beta, state,
        )
