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


class AddSameShapeTest(WorkloadBase):

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        b = torch.randn(self.n_total, dtype=self.dtype, device="cuda")
        return a, b


class DropoutWorkload(WorkloadBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, p: float = 0.5):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.p = p

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


class UnaryManifestWorkload:
    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


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


class UnaryBenchCase:
    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.randn(self.shape, device="cuda", dtype=self.dtype),)


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
    """ShapeDtypeWorkload for the generative ops (no input tensors)."""

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
