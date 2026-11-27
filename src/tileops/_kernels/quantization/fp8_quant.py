import dataclasses
from abc import abstractmethod

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.constants import FP8_E4M3_MAX
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._kernels.quantization.group_quant import (
    LAUNCH_CONFIG,
    exponent_scale,
    float32_constant,
    launch_regime,
    launch_shape,
    pow2_ceil_exponent,
    register_refusal,
    store_groups,
    write_in_place,
)

__all__ = ["FP8QuantCall", "FP8QuantFwdInterface", "FP8QuantKernel"]


@dataclasses.dataclass(frozen=True)
class FP8QuantCall(CallSpec):
    """One per-group FP8 quantization of a ``[*S, k]`` tensor, groups of ``group_size`` columns.

    The leading axes ``*S`` are not part of the call: one kernel serves any token count.
    """

    k: int = 0
    group_size: int = 0
    eps: float = 1e-10
    dtype: torch.dtype = torch.bfloat16
    column_major_scales: bool = False
    tma_aligned_scales: bool = False
    use_ue8m0: bool | None = None
    scale_dtype: torch.dtype = torch.float32
    inplace: bool = False


class FP8QuantFwdInterface(KernelInterface):
    """Group-wise quantization to ``float8_e4m3fn``."""

    request = FP8QuantCall

    @abstractmethod
    def forward(
        self, input_tensor: torch.Tensor, out_q: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Scale each group by its own maximum, writing the codes into *out_q* when given.

        A group is ``call.group_size`` adjacent columns of the last axis. Its scale is the
        absolute maximum over the group, floored at ``call.eps``, divided by 448; each element
        is multiplied by the reciprocal of that scale and clamped to the ``float8_e4m3fn``
        range. With ``call.use_ue8m0`` the scale is ``2 ** ceil(log2(m * r))``, ``m`` the
        floored maximum and ``r`` the float32 nearest ``1 / 448``, and each code is
        ``x / scale`` clamped and rounded to nearest even, exactly. A group holding an
        infinity or a NaN, or one whose floored maximum is below ``448 / 3.4e38`` so that the
        reciprocal overflows float32, has no defined result: whether the maximum and the
        clamp carry the non-finite value is left to the implementation. Nothing but *out_q*,
        or with ``call.inplace`` *input_tensor*, is written in place, and no output aliases
        the input.

        Args:
            input_tensor: ``[*S, call.k]`` in ``call.dtype`` on ``call.device``, contiguous;
                with ``call.inplace``, of any layout.
            out_q: ``[*S, call.k]`` ``float8_e4m3fn`` on ``call.device``, contiguous, the
                codes are stored into; or None.

        Returns:
            ``(q, scale)`` on ``call.device``: the codes, *out_q* where given and otherwise a
            new contiguous ``float8_e4m3fn`` tensor shaped as the input, and new contiguous
            ``call.scale_dtype`` ``[*S, call.k // call.group_size]`` scales. With
            ``call.inplace``, *input_tensor* is overwritten with each element's code times its
            scale and ``(None, None)`` returns.
        """


def _load_groups(block_m: int, group_size: int, predicated: bool):
    """A macro loading the groups at column ``col`` of rows ``first`` onward of ``rows`` into a
    ``[block_m, group_size]`` fragment.

    With *predicated*, the last block loads no row past ``rows``, for a program built with
    ``TL_ENABLE_LOWER_LDGSTG_PREDICATED``; otherwise it re-reads the last row there.
    """

    @T.macro
    def load_groups(input_tensor, first, rows, col, input_local):
        # Only the last block can reach past the rows; every other block loads without the
        # bound, which a runtime row count would otherwise cost each access.
        if first + block_m <= rows:
            for i, j in T.Parallel(block_m, group_size):
                input_local[i, j] = input_tensor[first + i, col + j]
        elif predicated:
            # The zeros in place of the rows past the last reach no store.
            for i, j in T.Parallel(block_m, group_size):
                input_local[i, j] = T.if_then_else(
                    first + i < rows, input_tensor[first + i, col + j], T.cast(0, input_local.dtype)
                )
        else:
            # The last block re-reads its last row; the stores drop those rows.
            for i, j in T.Parallel(block_m, group_size):
                input_local[i, j] = input_tensor[T.min(first + i, rows - 1), col + j]

    return load_groups


def _group_scales(block_m: int, eps: float, ue8m0: bool, amax_dtype: str):
    """A macro writing each loaded group's scale and the reciprocal its elements are
    multiplied by.

    The group maximum is reduced in *amax_dtype*: an absolute value and a maximum are exact
    in any float format, and a 16-bit one reduces two elements an instruction.
    """
    # act_quant multiplies by the float32 reciprocal of 448 rather than dividing.
    inv_fp8_max = float32_constant(1.0 / FP8_E4M3_MAX)

    @T.macro
    def group_scales(input_local, scale_local, recip_local):
        amax_local = T.alloc_fragment((block_m,), amax_dtype)
        T.reduce_absmax(input_local, amax_local, dim=1)
        for i in T.Parallel(block_m):
            amax = T.max(T.cast(amax_local[i], "float32"), eps)
            if ue8m0:
                # A product below 2 ** -126 takes 2 ** -126, as act_quant gives a subnormal
                # one; the floor also covers a zero product, which eps=0 admits.
                biased = T.max(pow2_ceil_exponent(amax * inv_fp8_max), T.uint32(1))
                scale_local[i] = exponent_scale(biased)
                recip_local[i] = exponent_scale(T.uint32(254) - biased)
            else:
                scale_local[i] = amax / FP8_E4M3_MAX
                recip_local[i] = 1.0 / scale_local[i]

    return group_scales


def _ue8m0_codes(block_m: int, group_size: int, in_dtype: str):
    """A macro writing each loaded element times its group's power-of-two reciprocal, as e4m3.

    The product is exact, so no rounding precedes the one to e4m3; in bfloat16, whose
    exponent range is float32's, it takes two elements an instruction. No clamp precedes the
    conversion: the scale is at least the group maximum times the float32 nearest 1 / 448,
    which exceeds 1 / 448, rounded to float32 and then up to a power of two, so a product is
    at most 448 * (1 + 6e-8), which rounds to 448 as a clamped one does. A scale raised to
    the 2 ** -126 floor only makes the products smaller.
    """
    # float16 cannot hold the reciprocal of a scale below 2 ** -15.
    product_dtype = in_dtype if in_dtype == "bfloat16" else "float32"

    @T.macro
    def ue8m0_codes(input_local, recip_local, output_local):
        recip_cast = T.alloc_fragment((block_m,), product_dtype)
        for i in T.Parallel(block_m):
            recip_cast[i] = T.cast(recip_local[i], product_dtype)
        for i, j in T.Parallel(block_m, group_size):
            output_local[i, j] = T.cast(input_local[i, j], product_dtype) * recip_cast[i]

    return ue8m0_codes


@target_cache(maxsize=32)
def _fp8_quant_kernel(
    group_size: int, in_dtype: str, threads: int, eps: float, ue8m0: bool, e8m0_scales: bool
):
    @tilelang.jit
    def _fp8_quant_fwd_func(block_m):
        out_dtype = T.float8_e4m3fn
        # An e8m0 scale is the exponent byte of its power-of-two float32 scale.
        scale_dtype = T.uint8 if e8m0_scales else T.float32
        load_groups = _load_groups(block_m, group_size, False)
        group_scales = _group_scales(block_m, eps, ue8m0, in_dtype)
        ue8m0_codes = _ue8m0_codes(block_m, group_size, in_dtype)
        store = store_groups(block_m, group_size, lambda scale, i: scale[i])
        # The group count is a runtime extent, so a new token count builds no program.
        rows = T.dynamic("rows")

        @T.prim_func
        def _fp8_quant_fwd_main(
            input_tensor: T.Tensor[(rows, group_size), in_dtype],
            scale_tensor: T.Tensor[(rows,), scale_dtype],
            output_tensor: T.Tensor[(rows, group_size), out_dtype],
        ):
            with T.Kernel(T.ceildiv(rows, block_m), threads=threads) as pid_m:
                input_local = T.alloc_fragment((block_m, group_size), in_dtype)
                output_local = T.alloc_fragment((block_m, group_size), out_dtype)
                scale_local = T.alloc_fragment((block_m,), "float32")
                recip_local = T.alloc_fragment((block_m,), "float32")
                scale_out = T.alloc_fragment((block_m,), scale_dtype)

                first = pid_m * block_m
                load_groups(input_tensor, first, rows, 0, input_local)
                group_scales(input_local, scale_local, recip_local)
                for i in T.Parallel(block_m):
                    if e8m0_scales:
                        scale_out[i] = T.cast(
                            T.reinterpret(scale_local[i], "uint32") >> T.uint32(23), scale_dtype
                        )
                    else:
                        scale_out[i] = scale_local[i]

                if ue8m0:
                    ue8m0_codes(input_local, recip_local, output_local)
                else:
                    # Not the reference's divide by the scale; FP8QuantPerTokenGroupFwdOp
                    # bounds the gap. No clamp: |x| is at most the floored maximum, whose
                    # product with the reciprocal is within a few float32 ulps of 448, so a
                    # product rounds to at most 448, as a clamped one does.
                    for i, j in T.Parallel(block_m, group_size):
                        output_local[i, j] = input_local[i, j] * recip_local[i]

                store(scale_tensor, output_tensor, first, rows, scale_out, output_local)

        return _fp8_quant_fwd_main

    return _fp8_quant_fwd_func


@target_cache(maxsize=32)
def _fp8_fake_quant_kernel(
    group_size: int, in_dtype: str, threads: int, eps: float, row_groups: int, row_stride: int
):
    # Lowers a bounded load to one predicated load, so a partial block's threads past the last
    # row issue no load and take no branch. Its inline assembly costs every load of the program
    # the read-only data path, which this program, writing its input, never has.
    predicated_loads = {tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True}

    @tilelang.jit(pass_configs=predicated_loads)
    def _fp8_fake_quant_fwd_func(block_m):
        load_groups = _load_groups(block_m, group_size, True)
        group_scales = _group_scales(block_m, eps, True, in_dtype)
        ue8m0_codes = _ue8m0_codes(block_m, group_size, in_dtype)
        width = row_groups * group_size
        rows = T.dynamic("rows")

        @T.prim_func
        def _fp8_fake_quant_fwd_main(
            input_tensor: T.StridedTensor[(rows, width), (row_stride, 1), in_dtype],
        ):
            # A block owns one group column of block_m rows.
            with T.Kernel(T.ceildiv(rows, block_m), row_groups, threads=threads) as (
                pid_m,
                pid_n,
            ):
                input_local = T.alloc_fragment((block_m, group_size), in_dtype)
                code_local = T.alloc_fragment((block_m, group_size), T.float8_e4m3fn)
                output_local = T.alloc_fragment((block_m, group_size), in_dtype)
                scale_local = T.alloc_fragment((block_m,), "float32")
                recip_local = T.alloc_fragment((block_m,), "float32")

                first = pid_m * block_m
                col = pid_n * group_size
                load_groups(input_tensor, first, rows, col, input_local)
                group_scales(input_local, scale_local, recip_local)
                ue8m0_codes(input_local, recip_local, code_local)
                # A code times its power-of-two scale is exact in float32; the cast to the
                # input dtype rounds as the reference's copy does, which is exact for
                # bfloat16 and float32 and for float16 above its subnormals.
                for i, j in T.Parallel(block_m, group_size):
                    output_local[i, j] = T.cast(
                        T.cast(code_local[i, j], "float32") * scale_local[i], in_dtype
                    )

                if first + block_m <= rows:
                    for i, j in T.Parallel(block_m, group_size):
                        input_tensor[first + i, col + j] = output_local[i, j]
                else:
                    for i, j in T.Parallel(block_m, group_size):
                        if first + i < rows:
                            input_tensor[first + i, col + j] = output_local[i, j]

        return _fp8_fake_quant_fwd_main

    return _fp8_fake_quant_fwd_func


class FP8QuantKernel(Kernel, FP8QuantFwdInterface):
    """Per-group fp8 quantization; one program per regime serves any number of groups.
    Fake quantization in place writes evenly spaced rows where they lie and any other layout
    through a contiguous copy.

    Args:
        k: Columns of the last axis.
        group_size: Elements per group; divides ``k``.
        eps: Floor of each group's absolute maximum.
        dtype: Torch dtype of the tensor being quantized.
        use_ue8m0: Round each scale up to a power of two.
        scale_dtype: ``torch.float32``, or ``torch.float8_e8m0fnu`` under ``use_ue8m0``.
        inplace: Overwrite the input with its fake-quantized values.
        config: Optional dict with the threads and 16-byte vectors a thread of the lane
            layout runs and loads in each launch regime, ``LAUNCH_CONFIG``'s keys.
    """

    supported_archs: list[int] = [89, 90]

    @classmethod
    def refusal(cls, call: FP8QuantCall) -> str | None:
        if call.column_major_scales:
            return "does not serve column-major scales"
        reason = register_refusal("group_size", call.group_size, call.dtype.itemsize)
        if reason is not None:
            return reason
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FP8QuantCall) -> Entry:
        return call, lambda: cls(
            call.k,
            call.group_size,
            call.eps,
            call.dtype,
            bool(call.use_ue8m0),
            call.scale_dtype,
            call.inplace,
        )

    def __init__(
        self,
        k: int,
        group_size: int,
        eps: float,
        dtype: torch.dtype,
        use_ue8m0: bool = False,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
        config: dict | None = None,
    ):
        super().__init__()
        self.k = k
        self.group_size = group_size
        self.eps = eps
        self.dtype = dtype
        self.use_ue8m0 = use_ue8m0
        self.scale_dtype = scale_dtype
        self.e8m0 = scale_dtype == torch.float8_e8m0fnu
        self.inplace = inplace
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return dict(LAUNCH_CONFIG)

    def forward(
        self, input_tensor: torch.Tensor, out_q: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        regime = launch_regime(input_tensor)
        if self.inplace:
            write_in_place(
                input_tensor,
                self.k,
                self.group_size,
                lambda rows, row_groups, row_stride: self.program(
                    self._compile_fake, regime, row_groups, row_stride
                )(rows),
            )
            return None, None
        groups = input_tensor.numel() // self.group_size
        q = torch.empty_like(input_tensor, dtype=torch.float8_e4m3fn) if out_q is None else out_q
        scale = torch.empty(
            (*input_tensor.shape[:-1], self.k // self.group_size),
            dtype=self.scale_dtype,
            device=input_tensor.device,
        )
        self.program(self._compile, regime)(
            input_tensor.view(groups, self.group_size),
            (scale.view(torch.uint8) if self.e8m0 else scale).view(groups),
            q.view(groups, self.group_size),
        )
        return q, scale

    def _compile(self, cfg: dict, regime: str):
        """The program of config *cfg* for a launch in *regime*."""
        threads, block_m = launch_shape(cfg, regime, self.group_size, self.dtype.itemsize)
        return _fp8_quant_kernel(
            self.group_size, self.dtype_str, threads, self.eps, self.use_ue8m0, self.e8m0
        )(block_m)

    def _compile_fake(self, cfg: dict, regime: str, row_groups: int, row_stride: int):
        """The in-place program of config *cfg* in *regime* over rows of *row_groups* groups,
        *row_stride* elements apart."""
        threads, block_m = launch_shape(cfg, regime, self.group_size, self.dtype.itemsize)
        return _fp8_fake_quant_kernel(
            self.group_size, self.dtype_str, threads, self.eps, row_groups, row_stride
        )(block_m)
