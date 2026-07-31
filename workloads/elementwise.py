"""Workload definitions for elementwise op workloads with custom generators."""

from math import prod
from typing import Callable, Optional

import torch

from workloads.workload_base import WorkloadBase


class ReluTest(WorkloadBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return (x,)
class BinaryManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        other_shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        positive: bool = False,
        integer: bool = False,
        logical: bool = False,
    ):
        self.input_shape = input_shape
        self.other_shape = other_shape
        self.a_shape = input_shape
        self.b_shape = other_shape
        self.shape = tuple(torch.broadcast_shapes(input_shape, other_shape))
        self.n_total = prod(self.shape)
        self.dtype = dtype
        self.positive = positive
        self.integer = integer
        self.logical = logical

    def _tensor(self, shape: tuple[int, ...]) -> torch.Tensor:
        if self.dtype is torch.bool:
            return torch.randint(0, 2, shape, device="cuda", dtype=torch.bool)
        if self.integer:
            return torch.randint(-1000, 1000, shape, device="cuda", dtype=self.dtype)
        if self.positive:
            return torch.rand(shape, device="cuda", dtype=self.dtype) + 0.1
        if self.logical:
            return (torch.randn(shape, device="cuda", dtype=self.dtype) > 0).to(self.dtype)
        return torch.randn(shape, device="cuda", dtype=self.dtype)

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tensor(self.input_shape), self._tensor(self.other_shape)


class PreluManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        weight_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = input_shape
        self.weight_shape = weight_shape
        self.shape = input_shape
        self.n_total = prod(input_shape)
        self.dtype = dtype

    @property
    def num_channels(self) -> int:
        return self.weight_shape[0] if self.weight_shape else 1

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device="cuda", dtype=self.dtype)
        return x, weight


class MaskedFillTensorManifestWorkload:
    def __init__(
        self,
        input_shape: tuple[int, ...],
        mask_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.value_shape = value_shape
        self.shape = tuple(torch.broadcast_shapes(input_shape, mask_shape))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.mask_shape, device="cuda") > 0.5
        value = torch.full(self.value_shape, -100.0, device="cuda", dtype=self.dtype)
        return x, mask, value


class MaskedFillScalarManifestWorkload:
    def __init__(self, input_shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = input_shape
        self.mask_shape = input_shape
        self.shape = input_shape
        self.n_total = prod(input_shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        mask = torch.rand(self.mask_shape, device="cuda") > 0.5
        return x, mask


class WhereManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.condition_shape = shape
        self.input_shape = shape
        self.other_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cond = torch.rand(self.condition_shape, device="cuda") > 0.5
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        y = torch.randn(self.other_shape, device="cuda", dtype=self.dtype)
        return cond, x, y


class LerpTensorManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.input_shape = shape
        self.end_shape = shape
        self.weight_shape = shape
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        end = torch.randn(self.end_shape, device="cuda", dtype=self.dtype)
        weight = torch.rand(self.weight_shape, device="cuda", dtype=self.dtype)
        return x, end, weight
class TensorClampBenchCase:
    """Workload adapter for Tensor-bound clamp ops.

    Holds the post-broadcast output shape so :class:`ManifestBenchmark`
    can read a single ``n_total`` while the bench builds per-operand
    tensors from the manifest-declared ``input_shape`` / ``min_shape`` /
    ``max_shape`` keys.
    """

    def __init__(
        self,
        input_shape: tuple,
        dtype: torch.dtype,
        min_shape: Optional[tuple] = None,
        max_shape: Optional[tuple] = None,
    ):
        self.input_shape = input_shape
        self.min_shape = min_shape
        self.max_shape = max_shape
        broadcast_args = [input_shape]
        if min_shape is not None:
            broadcast_args.append(min_shape)
        if max_shape is not None:
            broadcast_args.append(max_shape)
        self.shape = tuple(torch.broadcast_shapes(*broadcast_args))
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.input_shape, device="cuda", dtype=self.dtype)
        tensors: list[torch.Tensor] = [x]
        if self.min_shape is not None:
            tensors.append(
                torch.randn(self.min_shape, device="cuda", dtype=self.dtype) - 0.5
            )
        if self.max_shape is not None:
            tensors.append(
                torch.randn(self.max_shape, device="cuda", dtype=self.dtype) + 0.5
            )
        return tuple(tensors)


class _GenerativeWorkload:
    """Shape/dtype descriptor for the generative ops (no input tensors)."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple:
        return ()


class Fp8UnaryBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0
        return (x.to(self.dtype),)


class Fp8WhereBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        cond = torch.rand(self.shape, device="cuda") > 0.5
        x = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        y = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        return cond, x, y


class Fp8MaskedFillBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x = (torch.randn(self.shape, device="cuda", dtype=torch.float16) * 2.0).to(
            self.dtype
        )
        mask = torch.rand(self.shape, device="cuda") > 0.5
        return x, mask


class BinaryBenchCase:
    """Minimal workload for binary ops."""

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        gen_inputs: Callable,
    ):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._gen_inputs = gen_inputs

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gen_inputs(self.shape, self.dtype)


class FusedGatedBenchCase:
    """Minimal workload for fused gated ops."""

    def __init__(self, M: int, N: int, dtype: torch.dtype):
        self.M = M
        self.N = N
        self.n_total = M * N
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.M, 2 * self.N, device="cuda", dtype=self.dtype),)


class BroadcastBenchCase:
    """Workload for broadcast binary ops with asymmetric shapes."""

    def __init__(
        self,
        a_shape: tuple,
        b_shape: tuple,
        dtype: torch.dtype,
        output_dtype: torch.dtype,
        gen_inputs: Callable,
    ):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.n_total = prod(a_shape)  # output size = broadcast result
        self.dtype = dtype
        self.output_dtype = output_dtype
        self._gen_inputs = gen_inputs

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gen_inputs(self.a_shape, self.b_shape, self.dtype)
class AddBroadcastWorkload(WorkloadBase):
    def __init__(self, a_shape: tuple, b_shape: tuple, dtype: torch.dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b
class PowPositiveWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.5
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") * 2.0
        return a, b
class BitwiseNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
        elif self.dtype == torch.uint8:
            x = torch.randint(0, 256, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randint(-128, 128, (self.n_total,), device="cuda", dtype=self.dtype)
        return (x,)
class AddCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = torch.randn(self.b_shape, dtype=self.dtype, device="cuda")
        return a, b


class EqCompileWorkload(WorkloadBase):
    def __init__(self, a_shape, b_shape, dtype):
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.dtype = dtype

    def gen_inputs(self):
        a = torch.randn(self.a_shape, dtype=self.dtype, device="cuda")
        b = a.clone()
        mask = torch.rand_like(a, dtype=torch.float32) > 0.5
        b[mask] = torch.randn_like(b[mask])
        return a, b


class SiluAndMulCompileWorkload(WorkloadBase):
    def __init__(self, M, N, dtype):
        self.M = M
        self.N = N
        self.dtype = dtype

    def gen_inputs(self):
        x = torch.randn(self.M, 2 * self.N, dtype=self.dtype, device="cuda")
        return (x,)
class LogicalNotWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype == torch.bool:
            x = torch.rand(self.n_total, device="cuda") > 0.5
            return (x,)

        if self.dtype == torch.uint8:
            x = torch.randint(0, 8, (self.n_total,), device="cuda", dtype=self.dtype)
        elif self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            x = torch.randint(-4, 4, (self.n_total,), device="cuda", dtype=self.dtype)
        else:
            x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)

        mask = torch.rand(self.n_total, device="cuda") > 0.5
        x[mask] = 0
        return (x,)
class BitwiseWorkload(WorkloadBase):
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.dtype = torch.int32

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        b = torch.randint(-1000, 1000, (self.n_total,), dtype=torch.int32, device="cuda")
        return a, b
class LogicalWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda") > 0
        a = a.to(self.dtype)
        b = b.to(self.dtype)
        return a, b


class SpecialWorkload(WorkloadBase):
    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        quarter = self.n_total // 4
        x[:quarter] = float("nan")
        x[quarter:2 * quarter] = float("inf")
        x[2 * quarter:3 * quarter] = float("-inf")
        return (x,)


class RandnFlatWorkload(WorkloadBase):
    """One ``randn`` vector of ``n_total`` elements.

    ``gen_fn`` lets a caller substitute a domain-restricted draw (positive-only,
    NaN-seeded, ...) without another class.
    """

    def __init__(self, n_total: int, dtype: torch.dtype, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)


class ShapedRandnWorkload(WorkloadBase):
    """One ``randn`` tensor of arbitrary rank, with its element count."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = tuple(shape)
        self.n_total = prod(self.shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


class RandnPairWorkload(WorkloadBase):
    """Two same-shape ``randn`` vectors — the default binary-op input."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b


class PositivePairWorkload(WorkloadBase):
    """Two same-shape vectors in ``[0.1, 1.1)`` — for ops undefined at or below 0."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        b = torch.rand(self.n_total, dtype=self.dtype, device="cuda") + 0.1
        return a, b


class GatedRandnWorkload(WorkloadBase):
    """One ``(m, 2 * n)`` tensor — gate and value halves for a fused gated op."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.m, 2 * self.n, dtype=self.dtype, device="cuda"),)


# Value domains. A benchmark or test names the domain its op requires;
# the draw itself belongs to this layer so both stages get the same tensors.


def draw_normal_pair(shape: tuple, dtype: torch.dtype):
    a = torch.randn(*shape, device="cuda", dtype=dtype)
    b = torch.randn(*shape, device="cuda", dtype=dtype)
    return a, b


def draw_positive_pair(shape: tuple, dtype: torch.dtype):
    a = torch.rand(*shape, device="cuda", dtype=dtype) + 0.1
    b = torch.rand(*shape, device="cuda", dtype=dtype) + 0.1
    return a, b


def draw_int_pair(shape: tuple, dtype: torch.dtype):
    a = torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32)
    b = torch.randint(-1000, 1000, shape, device="cuda", dtype=torch.int32)
    return a, b


def draw_bool_pair(shape: tuple, dtype: torch.dtype):
    a = (torch.randn(*shape, device="cuda", dtype=dtype) > 0).to(dtype)
    b = (torch.randn(*shape, device="cuda", dtype=dtype) > 0).to(dtype)
    return a, b


def draw_normal_broadcast_pair(a_shape, b_shape, dtype):
    a = torch.randn(*a_shape, device="cuda", dtype=dtype)
    b = torch.randn(*b_shape, device="cuda", dtype=dtype)
    return a, b


def draw_positive_broadcast_pair(a_shape, b_shape, dtype):
    a = torch.rand(*a_shape, device="cuda", dtype=dtype) + 0.1
    b = torch.rand(*b_shape, device="cuda", dtype=dtype) + 0.1
    return a, b


def draw_normal(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    return (torch.randn(shape, device="cuda", dtype=dtype),)


def draw_positive_away_from_zero(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    # Domain restriction for log / sqrt / rsqrt / log1p / reciprocal.
    return (torch.rand(shape, device="cuda", dtype=dtype) + 0.5,)


def draw_bool(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, device="cuda", dtype=torch.bool)
    else:
        x = torch.randn(shape, device="cuda", dtype=dtype)
        mask = torch.rand(shape, device="cuda") > 0.5
        x[mask] = 0
    return (x,)


def draw_int(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    info = torch.iinfo(dtype)
    lo = max(info.min, -1024)
    hi = min(info.max, 1024)
    return (torch.randint(lo, hi, shape, device="cuda", dtype=dtype),)


def draw_special_floats(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    # Mix of normal floats, +/-inf, and NaN — exercises isnan/isinf/isfinite.
    x = torch.randn(shape, device="cuda", dtype=dtype)
    flat = x.view(-1)
    quarter = flat.numel() // 4
    flat[:quarter] = float("nan")
    flat[quarter:2 * quarter] = float("inf")
    flat[2 * quarter:3 * quarter] = float("-inf")
    return (x,)
