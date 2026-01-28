import itertools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["mhc_pre_kernel"]


def _mhc_pre_kernel(batch: int, n_expand: int, c_x: int, x_dtype: str = 'bfloat16'):

    def sigmoid(x):
        return 1 / (1 + T.exp2(-x * 1.44269504))

    dtype = "float32"
    accum_dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mhc_func(block_x_b, block_C, num_stages, threads=128):
        x_dim = n_expand * c_x
        phi_dim = n_expand * n_expand + 2 * n_expand
        block_x_b = min(batch, block_x_b)

        @T.macro
        def _get_H_0_no_split(
                phi: T.Tensor([x_dim, phi_dim], dtype),
                x: T.Tensor([batch, x_dim], x_dtype),
                r: T.Tensor([batch], dtype),
                H: T.Tensor([batch, phi_dim], dtype),
        ):

            with T.Kernel(batch // block_x_b, threads=threads) as (bx):

                phi_dim = n_expand * n_expand + 2 * n_expand
                x_shared = T.alloc_shared([block_x_b, block_C], x_dtype)
                phi_shared = T.alloc_shared([block_C, phi_dim], dtype)

                loop_range = T.ceildiv(n_expand * c_x, block_C)

                acc_x_phi = T.alloc_fragment([block_x_b, phi_dim], accum_dtype)
                xsqr = T.alloc_fragment([block_x_b, block_C], accum_dtype)
                acc_r_sqr = T.alloc_fragment([block_x_b], accum_dtype)
                xsqr_sum = T.alloc_fragment([block_x_b], accum_dtype)

                for i, j in T.Parallel(block_x_b, phi_dim):
                    acc_x_phi[i, j] = 0

                for i in T.Parallel(block_x_b):
                    acc_r_sqr[i] = 0

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(x[bx * block_x_b:(bx + 1) * block_x_b, k * block_C:(k + 1) * block_C],
                           x_shared)
                    # TODO: <*important> try to figure out why "T.copy" does not work...
                    # (probably a mbarrier failure)
                    # T.copy(phi[k * block_C : (k + 1) * block_C, :], phi_shared)
                    for i, j in T.Parallel(block_C, phi_dim):
                        phi_shared[i, j] = phi[k * block_C + i, j]
                    # When the batch size is smaller than 16, wgmma cannot be used... :(
                    for i, j in T.Parallel(block_x_b, phi_dim):
                        for iter in T.Serial(block_C):
                            acc_x_phi[i, j] += x_shared[i, iter] * phi_shared[iter, j]

                    for i, j in T.Parallel(block_x_b, block_C):
                        xsqr[i, j] += x_shared[i, j] * x_shared[i, j]
                T.reduce_sum(xsqr, xsqr_sum, 1)
                for i in T.Parallel(block_x_b):
                    acc_r_sqr[i] = T.sqrt(xsqr_sum[i])

                T.copy(acc_x_phi, H[bx * block_x_b:(bx + 1) * block_x_b, :])

                for i in T.Parallel(block_x_b):
                    acc_r_sqr[i] /= (n_expand * c_x)**0.5 + 0.0001
                T.copy(acc_r_sqr, r[bx * block_x_b:(bx + 1) * block_x_b])

        @T.macro
        def _get_H_1(
                H: T.Tensor([batch, n_expand * n_expand + 2 * n_expand], dtype),
                r: T.Tensor([batch], dtype),
                b: T.Tensor([n_expand * n_expand + 2 * n_expand], dtype),
                alpha_pre: T.float,
                alpha_post: T.float,
                alpha_res: T.float,
                H_pre: T.Tensor([batch, n_expand], dtype),
                H_post: T.Tensor([batch, n_expand], dtype),
                H_res_0: T.Tensor([batch, n_expand, n_expand], dtype),
        ):
            with T.Kernel(batch // block_x_b, threads=threads) as (bx):
                # needs binding
                h_pre_shared = T.alloc_shared([block_x_b, n_expand], dtype)
                h_post_shared = T.alloc_shared([block_x_b, n_expand], dtype)
                h_res_shared = T.alloc_shared([block_x_b, n_expand, n_expand], dtype)

                b_shared = T.alloc_shared([n_expand * n_expand + 2 * n_expand], dtype)
                h_res_frag = T.alloc_fragment([block_x_b, n_expand, n_expand], dtype)

                T.copy(H[bx * block_x_b:(bx + 1) * block_x_b, 0:n_expand], h_pre_shared)
                T.copy(H[bx * block_x_b:(bx + 1) * block_x_b, n_expand:2 * n_expand], h_post_shared)

                for i, j in T.Parallel(block_x_b, n_expand):
                    for k in T.Serial(n_expand):
                        h_res_shared[i, j, k] = H[bx * block_x_b + i,
                                                  2 * n_expand + j * n_expand + k]

                T.copy(h_res_shared, h_res_frag)

                # move b to shared memory...
                # TODO: Coalesced Memory Access
                for i in T.Parallel(n_expand * n_expand + 2 * n_expand):
                    b_shared[i] = b[i]

                for i, j in T.Parallel(block_x_b, n_expand * n_expand + 2 * n_expand):
                    if j < n_expand:
                        alpha = alpha_pre
                        h_pre_shared[i, j] = sigmoid(1 / r[bx * block_x_b + i] * alpha *
                                                     h_pre_shared[i, j] + b_shared[j])
                        H_pre[bx * block_x_b + i, j] = h_pre_shared[i, j]

                    elif j < 2 * n_expand:
                        alpha = alpha_post
                        h_post_shared[i, j -
                                      n_expand] = 2 * sigmoid(1 / r[bx * block_x_b + i] * alpha *
                                                              h_post_shared[i, j - n_expand] +
                                                              b_shared[j])
                        H_post[bx * block_x_b + i, j - n_expand] = h_post_shared[i, j - n_expand]

                    else:
                        alpha = alpha_res
                        j_tmp = j - 2 * n_expand
                        row_res = j_tmp // n_expand
                        col_res = j_tmp % n_expand
                        h_res_shared[i, row_res, col_res] = (
                            1 / r[bx * block_x_b + i] * alpha * h_res_shared[i, row_res, col_res] +
                            b_shared[j])
                        H_res_0[bx * block_x_b + i, row_res, col_res] = h_res_shared[i, row_res,
                                                                                     col_res]

        @T.macro
        def _get_H_res(
                H_res_0: T.Tensor([batch, n_expand, n_expand], dtype),
                sinkhorn_repeat: T.int,
                H_res: T.Tensor([batch, n_expand, n_expand], dtype),
        ):
            with T.Kernel(batch, threads=threads) as (bx):
                h_frag = T.alloc_fragment([n_expand, n_expand], dtype)
                T.copy(H_res_0[bx, :, :], h_frag)
                tmp1 = T.alloc_fragment([n_expand], dtype)
                tmp2 = T.alloc_fragment([n_expand], dtype)
                h_out_shared = T.alloc_shared([n_expand, n_expand], dtype)

                # exponential function...
                # get the max value first...
                for i, j in T.Parallel(n_expand, n_expand):
                    h_out_shared[i, j] = H_res_0[bx, i, j]

                for i, j in T.Parallel(n_expand, n_expand):
                    h_frag[i, j] = h_out_shared[i, j]
                T.reduce_max(h_frag, tmp1)

                for i, j in T.Parallel(n_expand, n_expand):
                    h_frag[i, j] = T.exp2((h_frag[i, j] - tmp1[i]) * 1.44269504)

                #for iter_sinkhorn in T.Pipelined(sinkhorn_repeat):
                for _iter_sinkhorn in T.Serial(sinkhorn_repeat):
                    T.reduce_sum(h_frag, tmp1, dim=1)
                    for j, k in T.Parallel(n_expand, n_expand):
                        h_frag[j, k] /= (tmp1[j] + 0.0001)
                    T.reduce_sum(h_frag, tmp2, dim=0)
                    for j, k in T.Parallel(n_expand, n_expand):
                        h_frag[j, k] /= (tmp2[k] + 0.0001)

                T.copy(h_frag, h_out_shared)
                for i, j in T.Parallel(n_expand, n_expand):
                    H_res[bx, i, j] = h_frag[i, j]

        @T.macro
        def _get_x(
                x_in: T.Tensor([batch, n_expand * c_x], dtype),
                h_pre: T.Tensor([batch, n_expand], dtype),
                h_res: T.Tensor([batch, n_expand, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_layer: T.Tensor([batch, c_x], x_dtype),
        ):
            with T.Kernel(batch, c_x // block_C, threads=threads) as (bx, by):

                h_pre_shared = T.alloc_shared([n_expand], dtype)
                h_res_shared = T.alloc_shared([n_expand, n_expand], dtype)

                x_in_shared = T.alloc_shared([n_expand, block_C], x_dtype)
                x_res_frag = T.alloc_fragment([n_expand, block_C], x_dtype)
                x_layer_frag = T.alloc_fragment([block_C], x_dtype)

                for i in T.Parallel(n_expand):
                    h_pre_shared[i] = h_pre[bx, i]

                for i, j in T.Parallel(n_expand, block_C):
                    x_in_shared[i, j] = x_in[bx, i * c_x + by * block_C + j]

                for i, j in T.Parallel(n_expand, n_expand):
                    h_res_shared[i, j] = h_res[bx, i, j]

                # calculate x_layer
                for i in T.Parallel(block_C):
                    x_layer_frag[i] = 0
                    for j in T.Serial(n_expand):
                        x_layer_frag[i] += h_pre_shared[j] * x_in_shared[j, i]

                    x_layer[bx, block_C * by + i] = x_layer_frag[i]

                # calculate x_res
                for i, j in T.Parallel(n_expand, block_C):
                    x_res_frag[i, j] = 0
                    for k in T.Serial(n_expand):
                        x_res_frag[i, j] += h_res_shared[i, k] * x_in_shared[k, j]

                    x_res[bx, i * c_x + by * block_C + j] = x_res_frag[i, j]

        @T.prim_func
        def mhc_pre(
                phi: T.Tensor([x_dim, phi_dim], dtype),
                x: T.Tensor([batch, x_dim], x_dtype),
                H: T.Tensor([batch, n_expand * n_expand + 2 * n_expand], dtype),
                r: T.Tensor([batch], dtype),
                b: T.Tensor([n_expand * n_expand + 2 * n_expand], dtype),
                alpha_pre: T.float,
                alpha_post: T.float,
                alpha_res: T.float,
                H_pre: T.Tensor([batch, n_expand], dtype),
                H_post: T.Tensor([batch, n_expand], dtype),
                H_res_0: T.Tensor([batch, n_expand, n_expand], dtype),
                sinkhorn_repeat: T.int,
                H_res: T.Tensor([batch, n_expand, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_layer: T.Tensor([batch, c_x], x_dtype),
        ):
            _get_H_0_no_split(phi, x, r, H)
            _get_H_1(H, r, b, alpha_pre, alpha_post, alpha_res, H_pre, H_post, H_res_0)
            _get_H_res(H_res_0, sinkhorn_repeat, H_res)
            _get_x(x, H_pre, H_res, x_res, x_layer)

        return mhc_pre

    return _mhc_func


@torch.library.custom_op("top::mhc_pre_wrapped_kernel", mutates_args=())
def _mhc_pre_wrapped_kernel(batch: int, n_expand: int, c_x: int, dtype: str, block_x_b: int,
                            block_C: int, num_stages: int, threads: int, phi: torch.Tensor,
                            x: torch.Tensor, H: torch.Tensor, r: torch.Tensor, b: torch.Tensor,
                            alpha_pre: float, alpha_post: float, alpha_res: float,
                            H_pre: torch.Tensor, H_post: torch.Tensor, H_res_0: torch.Tensor,
                            sinkhorn_repeat: int, H_res: torch.Tensor,
                            x_res: torch.Tensor) -> torch.Tensor:
    return _mhc_pre_kernel(batch, n_expand, c_x,
                           dtype)(block_x_b, block_C, num_stages,
                                  threads)(phi, x, H, r, b, alpha_pre, alpha_post, alpha_res, H_pre,
                                           H_post, H_res_0, sinkhorn_repeat, H_res, x_res)


@_mhc_pre_wrapped_kernel.register_fake
def _(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: str,
    num_stages: int,
    threads: int,
    *input,
) -> torch.Tensor:
    return torch.empty_like(input[0], dtype=input[0].dtype, device=input[0].device)


class mhc_pre_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: str = 'float32',
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.kernel = _mhc_pre_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {'block_x_b': 1, 'block_C': 64, "num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        num_stages = [2, 3]
        threads = [128, 256]
        block_x_b = [1, 8, 64]
        block_C = [64, 128]
        _configs = list(itertools.product(block_x_b, block_C, num_stages, threads))

        configs = [{
            'block_x_b': c[0],
            'block_C': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs

    def forward(self, phi, x, b, alpha_pre, alpha_post, alpha_res, sinkhorn_repeat):
        # H_pre, H_post, H_res_0, H_res are tensors need to be allocated....
        r = torch.randn([self.batch], device=x.device, dtype=self.weights_dtype)
        H = torch.randn([self.batch, self.n_expand * self.n_expand + 2 * self.n_expand],
                        device=x.device,
                        dtype=self.weights_dtype)
        H_pre = torch.randn((self.batch, self.n_expand), device=x.device, dtype=self.weights_dtype)
        H_res_0 = torch.randn([self.batch, self.n_expand, self.n_expand],
                              device=x.device,
                              dtype=self.weights_dtype)
        H_res = torch.randn([self.batch, self.n_expand, self.n_expand],
                            device=x.device,
                            dtype=self.weights_dtype)
        H_post = torch.randn((self.batch, self.n_expand), device=x.device, dtype=self.weights_dtype)
        x_res = torch.empty_like(x, device=x.device, dtype=x.dtype)

        result = _mhc_pre_wrapped_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str,
                                         self.config["block_x_b"], self.config["block_C"],
                                         self.config["num_stages"], self.config["threads"], phi, x,
                                         H, r, b, alpha_pre, alpha_post, alpha_res, H_pre, H_post,
                                         H_res_0, sinkhorn_repeat, H_res, x_res)
        return x_res, result


def main():
    batch = 4
    n_expand = 4
    c_x = 2560

    phi = torch.randn([n_expand * c_x, n_expand * n_expand + 2 * n_expand],
                      device="cuda",
                      dtype=torch.float32)
    x = torch.randn([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
    b = torch.randn([n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32)
    alpha_pre = torch.randn(())
    alpha_post = torch.randn(())
    alpha_res = torch.randn(())
    sinkhorn_repeat = 20
    test_mhc_kernel = mhc_pre_kernel(batch, n_expand, c_x, dtype=torch.bfloat16)

    x_res, x_layer = test_mhc_kernel.forward(phi, x, b, alpha_pre, alpha_post, alpha_res,
                                             sinkhorn_repeat)
    print(x_res)
    print(x_layer)

    # check the correctness with torch...
    xsqr = x * x  # the square of x
    r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x)
    H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float)
    for i in range(batch):
        H[i, :] = x[i, :].float() @ phi

    H_pre_ref = H[:, :n_expand]
    H_res_ref = H[:, 2 * n_expand:]
    H_res_ref = H_res_ref.reshape(batch, n_expand, n_expand)

    b_pre_ref = b[:n_expand]
    b_res_ref = b[2 * n_expand:]
    b_res_ref = b_res_ref.reshape([n_expand, n_expand])

    H_pre_ref = torch.sigmoid(alpha_pre * H_pre_ref / r_ref.unsqueeze(-1) + b_pre_ref)
    H_res_ref = alpha_res * H_res_ref / r_ref.unsqueeze(-1).unsqueeze(-1) + b_res_ref

    eps = 0.0001
    H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

    H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
    for _i in range(sinkhorn_repeat):
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
    x_in_reshaped = x.reshape([batch, n_expand, c_x])
    x_res_ref = torch.zeros([batch, n_expand, c_x], device="cuda", dtype=torch.bfloat16)
    x_layer_ref = torch.zeros([batch, c_x], device="cuda", dtype=torch.bfloat16)

    h_res_ref = H_res_ref
    h_pre_ref = H_pre_ref
    for i in range(batch):
        h_res_tmp = h_res_ref[i, :, :].float()
        h_pre_tmp = h_pre_ref[i, :].float()
        x_in_reshaped_tmp = x_in_reshaped[i, :, :].float()
        x_res_ref[i, :, :] = h_res_tmp @ x_in_reshaped_tmp
        x_layer_ref[i, :] = h_pre_tmp @ x_in_reshaped_tmp

    x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

    x_res_ref = x_res_ref.bfloat16()
    x_layer_ref = x_layer_ref.bfloat16()
    print(x_res_ref)
    print(x_layer_ref)
    print(x_res_ref - x_res)


if __name__ == "__main__":
    main()
