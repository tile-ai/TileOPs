import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.attention.online_softmax import LOG2E
from tileops.kernels.kernel_base import Kernel

__all__ = ["MHCPreKernel"]


@functools.lru_cache(maxsize=32)
def _mhc_pre_kernel(batch: int, n_expand: int, c_x: int, x_dtype: str = "bfloat16"):
    """Build the mHC pre step as two launches over a split-K workspace.

    ``_project`` gives each ``block_K`` slice of ``x @ phi`` its own CTA and writes that
    slice's partial products, plus its partial ``sum(x * x)`` in the last column, to
    ``partial``. ``_mix`` reduces those partials per batch row, derives ``H_pre`` and the
    Sinkhorn-normalised ``H_res``, and applies them to one ``block_C`` column tile of ``x``.
    """
    dtype = "float32"
    x_dim = n_expand * c_x
    phi_dim = n_expand * n_expand + 2 * n_expand
    res_off = 2 * n_expand

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _mhc_func(block_K, block_b, block_C):
        num_splits = T.ceildiv(x_dim, block_K)
        block_b = min(batch, block_b)

        @T.macro
        def _project(
            phi: T.Tensor([x_dim, phi_dim], dtype),
            x: T.Tensor([batch, x_dim], x_dtype),
            partial: T.Tensor([batch, num_splits, phi_dim + 1], dtype),
        ):
            with T.Kernel(num_splits, T.ceildiv(batch, block_b), threads=128) as (bk, bb):
                x_shared = T.alloc_shared([block_b, block_K], dtype)
                phi_shared = T.alloc_shared([block_K, phi_dim], dtype)
                acc = T.alloc_fragment([block_b, phi_dim + 1], dtype)

                for i, k in T.Parallel(block_b, block_K):
                    x_shared[i, k] = T.if_then_else(
                        bb * block_b + i < batch and bk * block_K + k < x_dim,
                        x[bb * block_b + i, bk * block_K + k],
                        0,
                    )
                for k, j in T.Parallel(block_K, phi_dim):
                    phi_shared[k, j] = T.if_then_else(
                        bk * block_K + k < x_dim, phi[bk * block_K + k, j], 0
                    )
                for i, j in T.Parallel(block_b, phi_dim + 1):
                    acc[i, j] = 0
                    for k in T.Serial(block_K):
                        if j < phi_dim:
                            acc[i, j] += x_shared[i, k] * phi_shared[k, j]
                        else:
                            acc[i, j] += x_shared[i, k] * x_shared[i, k]
                for i, j in T.Parallel(block_b, phi_dim + 1):
                    if bb * block_b + i < batch:
                        partial[bb * block_b + i, bk, j] = acc[i, j]

        @T.macro
        def _mix(
            partial: T.Tensor([batch, num_splits, phi_dim + 1], dtype),
            x: T.Tensor([batch, x_dim], x_dtype),
            b: T.Tensor([phi_dim], dtype),
            alpha_pre: T.float,
            alpha_res: T.float,
            sinkhorn_repeat: T.int,
            sinkhorn_eps: T.float,
            x_res: T.Tensor([batch, x_dim], x_dtype),
            x_layer: T.Tensor([batch, c_x], x_dtype),
        ):
            with T.Kernel(T.ceildiv(c_x, block_C), batch, threads=128) as (bc, bx):
                h_shared = T.alloc_shared([phi_dim + 1], dtype)
                h_res = T.alloc_fragment([n_expand, n_expand], dtype)
                row = T.alloc_fragment([n_expand], dtype)
                col = T.alloc_fragment([n_expand], dtype)
                h_res_shared = T.alloc_shared([n_expand, n_expand], dtype)
                h_pre_shared = T.alloc_shared([n_expand], dtype)
                x_shared = T.alloc_shared([n_expand, block_C], dtype)

                for i, c in T.Parallel(n_expand, block_C):
                    x_shared[i, c] = T.if_then_else(
                        bc * block_C + c < c_x, x[bx, i * c_x + bc * block_C + c], 0
                    )
                for j in T.Parallel(phi_dim + 1):
                    acc = T.alloc_var(dtype)
                    acc = 0
                    for s in T.Serial(num_splits):
                        acc += partial[bx, s, j]
                    h_shared[j] = acc

                inv_r = 1 / (T.sqrt(h_shared[phi_dim]) / x_dim**0.5 + 0.0001)
                for j in T.Parallel(n_expand):
                    h_pre_shared[j] = 1 / (
                        1 + T.exp2(-(alpha_pre * inv_r * h_shared[j] + b[j]) * LOG2E)
                    )
                for i, k in T.Parallel(n_expand, n_expand):
                    h_res[i, k] = (
                        alpha_res * inv_r * h_shared[res_off + i * n_expand + k]
                        + b[res_off + i * n_expand + k]
                    )
                T.reduce_max(h_res, row, dim=1)
                for i, k in T.Parallel(n_expand, n_expand):
                    h_res[i, k] = T.exp2((h_res[i, k] - row[i]) * LOG2E)
                for _ in T.Serial(sinkhorn_repeat):
                    T.reduce_sum(h_res, row, dim=1)
                    for i, k in T.Parallel(n_expand, n_expand):
                        h_res[i, k] /= row[i] + sinkhorn_eps
                    T.reduce_sum(h_res, col, dim=0)
                    for i, k in T.Parallel(n_expand, n_expand):
                        h_res[i, k] /= col[k] + sinkhorn_eps
                T.copy(h_res, h_res_shared)

                for c in T.Parallel(block_C):
                    acc = T.alloc_var(dtype)
                    acc = 0
                    for j in T.Serial(n_expand):
                        acc += h_pre_shared[j] * x_shared[j, c]
                    if bc * block_C + c < c_x:
                        x_layer[bx, bc * block_C + c] = acc
                for i, c in T.Parallel(n_expand, block_C):
                    acc = T.alloc_var(dtype)
                    acc = 0
                    for k in T.Serial(n_expand):
                        acc += h_res_shared[i, k] * x_shared[k, c]
                    if bc * block_C + c < c_x:
                        x_res[bx, i * c_x + bc * block_C + c] = acc

        @T.prim_func
        def mhc_pre(
            phi: T.Tensor([x_dim, phi_dim], dtype),
            x: T.Tensor([batch, x_dim], x_dtype),
            b: T.Tensor([phi_dim], dtype),
            alpha_pre: T.float,
            alpha_res: T.float,
            sinkhorn_repeat: T.int,
            sinkhorn_eps: T.float,
            partial: T.Tensor([batch, num_splits, phi_dim + 1], dtype),
            x_res: T.Tensor([batch, x_dim], x_dtype),
            x_layer: T.Tensor([batch, c_x], x_dtype),
        ):
            _project(phi, x, partial)
            _mix(partial, x, b, alpha_pre, alpha_res, sinkhorn_repeat, sinkhorn_eps, x_res, x_layer)

        return mhc_pre

    return _mhc_func


class MHCPreKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(
        self,
        batch,
        n_expand,
        c_x,
        dtype: torch.dtype = torch.float32,
        config: Optional[dict] = None,
        tune=False,
    ):
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.kernel = _mhc_pre_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str)

        self._supply_prog = self._make_supply_prog()
        self.init_config(config, tune)

    def _make_supply_prog(self):
        """Create a supply_prog that handles scalar parameters (alpha_*, sinkhorn_*)."""
        from tilelang.utils.tensor import get_tensor_supply as _get_tensor_supply

        default_supply = _get_tensor_supply(tilelang.TensorSupplyType.Auto)

        # Scalar defaults: alpha_pre, alpha_res are T.float;
        # sinkhorn_repeat is T.int; sinkhorn_eps is T.float
        scalar_defaults = {
            "int32": 20,  # sinkhorn_repeat
            "float32": 0.5,  # alpha or sinkhorn_eps
        }

        def supply_prog(params):
            inputs = []
            for param in params:
                if param.is_scalar():
                    inputs.append(scalar_defaults.get(str(param.dtype), 1))
                else:
                    inputs.append(default_supply(param))
            return inputs

        return supply_prog

    @property
    def autotune_supply_prog(self):
        return self._supply_prog

    @property
    def default_config(self) -> dict:
        return {"block_K": 128, "block_b": 4, "block_C": 256}

    @property
    def autotune_configs(self) -> list[dict]:
        return [
            {"block_K": k, "block_b": bb, "block_C": c}
            for k, bb, c in itertools.product([64, 128, 256], [4, 8], [128, 256])
        ]

    def forward(self, phi, x, b, alpha_pre, alpha_res, sinkhorn_repeat, sinkhorn_eps=0.02):
        num_splits = -(-self.n_expand * self.c_x // self.config["block_K"])
        partial = torch.empty(
            [self.batch, num_splits, self.n_expand * self.n_expand + 2 * self.n_expand + 1],
            device=x.device,
            dtype=self.weights_dtype,
        )
        x_res = torch.empty_like(x)
        x_layer = torch.empty([self.batch, self.c_x], device=x.device, dtype=x.dtype)
        self.kernel(self.config["block_K"], self.config["block_b"], self.config["block_C"])(
            phi, x, b, alpha_pre, alpha_res, sinkhorn_repeat, sinkhorn_eps, partial, x_res, x_layer
        )
        return x_res, x_layer
