import functools

import tilelang
import tilelang.language as T


def conv_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype_bytes: int,
) -> int:
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage_bytes * max(1, num_stages)


def _prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _compute_out_shape(
    in_spatial_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        (in_spatial_shape[i] + 2 * padding[i] - kernel_shape[i]) // stride[i] + 1
        for i in range(len(in_spatial_shape))
    )


@functools.lru_cache(maxsize=128)
def make_conv_nd_implicit_gemm_kernel(
    *,
    batch: int,
    c_in: int,
    c_out: int,
    in_spatial_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    has_bias: bool,
    dtype: str = "float16",
    enable_hopper_im2col: bool = False,
):
    ndim = len(in_spatial_shape)
    if ndim not in {1, 2, 3}:
        raise ValueError(f"Unsupported convolution rank: {ndim}")
    if not (len(kernel_shape) == len(stride) == len(padding) == ndim):
        raise ValueError("in_spatial_shape, kernel_shape, stride, and padding must have the same rank")

    accum_dtype = "float"
    out_spatial_shape = _compute_out_shape(in_spatial_shape, kernel_shape, stride, padding)
    out_spatial_size = _prod(out_spatial_shape)
    kernel_spatial_size = _prod(kernel_shape)
    k_total = kernel_spatial_size * c_in

    x_shape = (batch, *in_spatial_shape, c_in)
    weight_shape = (*kernel_shape, c_in, c_out)
    out_shape = (batch, *out_spatial_shape, c_out)

    @tilelang.jit(out_idx=[2], compile_flags=["-O3", "-DENABLE_BF16"])
    def _conv_nd_func(
        block_m: int,
        block_n: int,
        block_k: int,
        num_stages: int,
        threads: int,
        enable_rasteration: bool,
    ):
        @T.prim_func
        def _conv_nd_main(
            x: T.Tensor(x_shape, dtype),  # type: ignore
            weight: T.Tensor(weight_shape, dtype),  # type: ignore
            out: T.Tensor(out_shape, dtype),  # type: ignore
            bias: T.Tensor((c_out,), dtype),  # type: ignore
        ):
            use_hopper_im2col = (
                enable_hopper_im2col
                and ndim == 2
                and kernel_shape[0] == kernel_shape[1]
                and stride[0] == stride[1]
                and padding[0] == padding[1]
                and c_in % block_k == 0
            )

            total_m = batch * out_spatial_size

            with T.Kernel(
                T.ceildiv(c_out, block_n),
                T.ceildiv(total_m, block_m),
                threads=threads,
            ) as (bx, by):
                data_shared = T.alloc_shared((block_m, block_k), dtype)
                weight_shared = T.alloc_shared((block_k, block_n), dtype)
                out_local = T.alloc_fragment((block_m, block_n), accum_dtype)
                out_shared = T.alloc_shared((block_m, block_n), dtype)

                weight_flat = T.Tensor((k_total, c_out), dtype, weight.data)
                out_flat = T.Tensor((total_m, c_out), dtype, out.data)

                T.use_swizzle(10, enable=enable_rasteration)
                T.clear(out_local)

                for k_iter in T.Pipelined(T.ceildiv(k_total, block_k), num_stages=num_stages):
                    if use_hopper_im2col:
                        T.c2d_im2col(
                            x,
                            data_shared,
                            by,
                            k_iter,
                            kernel_shape[0],
                            stride[0],
                            1,
                            padding[0],
                        )
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            m_idx = by * block_m + i
                            k_idx = k_iter * block_k + j
                            in_bound = (m_idx < total_m) & (k_idx < k_total)

                            if ndim == 1:
                                kw = k_idx // c_in
                                ci = k_idx % c_in
                                batch_idx = m_idx // out_spatial_shape[0]
                                ol = m_idx % out_spatial_shape[0]
                                il = ol * stride[0] + kw - padding[0]
                                in_bound = in_bound & (il >= 0) & (il < in_spatial_shape[0])
                                data_shared[i, j] = T.if_then_else(
                                    in_bound,
                                    x[batch_idx, il, ci],
                                    T.cast(0.0, dtype),
                                )
                            elif ndim == 2:
                                kh = k_idx // (kernel_shape[1] * c_in)
                                kw = (k_idx // c_in) % kernel_shape[1]
                                ci = k_idx % c_in
                                out_idx = m_idx % out_spatial_size
                                batch_idx = m_idx // out_spatial_size
                                oh = out_idx // out_spatial_shape[1]
                                ow = out_idx % out_spatial_shape[1]
                                ih = oh * stride[0] + kh - padding[0]
                                iw = ow * stride[1] + kw - padding[1]
                                in_bound = (
                                    in_bound
                                    & (ih >= 0)
                                    & (iw >= 0)
                                    & (ih < in_spatial_shape[0])
                                    & (iw < in_spatial_shape[1])
                                )
                                data_shared[i, j] = T.if_then_else(
                                    in_bound,
                                    x[batch_idx, ih, iw, ci],
                                    T.cast(0.0, dtype),
                                )
                            else:
                                kd = k_idx // (kernel_shape[1] * kernel_shape[2] * c_in)
                                kh = (k_idx // (kernel_shape[2] * c_in)) % kernel_shape[1]
                                kw = (k_idx // c_in) % kernel_shape[2]
                                ci = k_idx % c_in
                                out_idx = m_idx % out_spatial_size
                                batch_idx = m_idx // out_spatial_size
                                od = out_idx // (out_spatial_shape[1] * out_spatial_shape[2])
                                oh = (out_idx // out_spatial_shape[2]) % out_spatial_shape[1]
                                ow = out_idx % out_spatial_shape[2]
                                id_ = od * stride[0] + kd - padding[0]
                                ih = oh * stride[1] + kh - padding[1]
                                iw = ow * stride[2] + kw - padding[2]
                                in_bound = (
                                    in_bound
                                    & (id_ >= 0)
                                    & (ih >= 0)
                                    & (iw >= 0)
                                    & (id_ < in_spatial_shape[0])
                                    & (ih < in_spatial_shape[1])
                                    & (iw < in_spatial_shape[2])
                                )
                                data_shared[i, j] = T.if_then_else(
                                    in_bound,
                                    x[batch_idx, id_, ih, iw, ci],
                                    T.cast(0.0, dtype),
                                )

                    T.copy(weight_flat[k_iter * block_k, bx * block_n], weight_shared)
                    T.gemm(data_shared, weight_shared, out_local)

                for i, j in T.Parallel(block_m, block_n):
                    m_idx = by * block_m + i
                    oc = bx * block_n + j
                    if has_bias:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < total_m) & (oc < c_out),
                            T.cast(out_local[i, j] + T.cast(bias[oc], accum_dtype), dtype),
                            T.cast(0.0, dtype),
                        )
                    else:
                        out_shared[i, j] = T.if_then_else(
                            (m_idx < total_m) & (oc < c_out),
                            T.cast(out_local[i, j], dtype),
                            T.cast(0.0, dtype),
                        )

                T.copy(out_shared, out_flat[by * block_m, bx * block_n])

        return _conv_nd_main

    return _conv_nd_func
