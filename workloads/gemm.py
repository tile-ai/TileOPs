"""Workloads for GEMM, batched matmul and grouped GEMM."""

from typing import Any

import torch

from workloads.device import run_device
from workloads.workload_base import WorkloadBase

W4A16_GROUP_SIZE = 128


_FP8_INIT_SCALE: float = 0.25


class GemmWorkload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        trans_a: bool = False,
        trans_b: bool = False,
    ):
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b

    @classmethod
    def from_call(cls, call: Any) -> "GemmWorkload":
        """The workload of one manifest call of ``GemmFwdOp``."""
        ix = call.ix
        return cls(ix["M"], ix["N"], ix["K"], getattr(torch, ix["T"]), ix["trans_a"], ix["trans_b"])

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        shape_a = (self.k, self.m) if self.trans_a else (self.m, self.k)
        a = torch.randn(*shape_a, device=run_device(), dtype=self.dtype)
        shape_b = (self.n, self.k) if self.trans_b else (self.k, self.n)
        b = torch.randn(*shape_b, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFp8Workload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        scale_mode: str,
        out_dtype: torch.dtype = torch.bfloat16,
        bias: bool = False,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.scale_mode = scale_mode
        self.out_dtype = out_dtype
        self.bias = bias

    @classmethod
    def from_call(cls, call: Any) -> "GemmFp8Workload":
        """The workload of one manifest call of ``GemmFp8FwdOp``."""
        ix = call.ix
        if tuple(ix["SA"]) == (1, 1):
            scale_mode = "per_tensor"
        elif tuple(ix["SB"]) == (ix["N"], -(-ix["K"] // 128)):
            scale_mode = "block128"
        else:
            scale_mode = "block128x128"
        return cls(
            ix["M"],
            ix["N"],
            ix["K"],
            getattr(torch, ix["T"]),
            scale_mode,
            out_dtype=getattr(torch, ix["out_dtype"]),
            bias=call.present("bias"),
        )

    def _scale_shapes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.scale_mode in ("per_tensor", "tensor"):
            return (1, 1), (1, 1)
        if self.scale_mode == "block128":
            if self.k % 128 != 0:
                raise ValueError("block128 FP8 workloads require k divisible by 128")
            return (self.m, self.k // 128), (self.n, self.k // 128)
        if self.scale_mode == "block128x128":
            if self.k % 128 != 0:
                raise ValueError("block128x128 FP8 workloads require k divisible by 128")
            return (self.m, self.k // 128), (-(-self.n // 128), self.k // 128)
        raise ValueError(f"unknown FP8 GEMM scale_mode {self.scale_mode!r}")

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (torch.randn(self.m, self.k, device=run_device()) * 0.25).to(self.dtype).contiguous()
        b = (torch.randn(self.n, self.k, device=run_device()) * 0.25).to(self.dtype).contiguous()
        scale_a_shape, scale_b_shape = self._scale_shapes()
        scale_a = (
            0.5 + torch.rand(*scale_a_shape, device=run_device(), dtype=torch.float32)
        ).contiguous()
        scale_b = (
            0.5 + torch.rand(*scale_b_shape, device=run_device(), dtype=torch.float32)
        ).contiguous()
        if self.bias:
            bias = torch.randn(self.n, device=run_device(), dtype=self.out_dtype)
            return a, b, scale_a, scale_b, bias
        return a, b, scale_a, scale_b

    def _expand_scale(self, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """One scale per element of the ``[rows, cols]`` operand it multiplies.

        A scale grid is per tensor, per 1x128 block along ``cols``, or per 128x128 block.
        """
        if tuple(scale.shape) == (1, 1):
            return scale.expand(rows, cols)
        scale_cols = (cols + 127) // 128
        if tuple(scale.shape) == (rows, scale_cols):
            return scale.repeat_interleave(128, dim=1)[:, :cols]
        if tuple(scale.shape) == ((rows + 127) // 128, scale_cols):
            return scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
        raise ValueError(f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(scale_a, self.m, self.k)
        b_f = b.float() * self._expand_scale(scale_b, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)


def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = W4A16_GROUP_SIZE,
    scale_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine group-wise quantize and pack a logical ``[N, K]`` weight tensor.

    Args:
        weight: Logical weight, $[N \\times K]$.
        group_size: How many K values one scale and zero point cover.
        scale_dtype: Storage dtype of the scale, which follows the activation
            dtype an INT4 checkpoint is served at.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    n, k = weight.shape
    if k % group_size != 0:
        raise ValueError(f"K must be divisible by group_size={group_size}, got {k}")
    if k % 2 != 0:
        raise ValueError(f"K must be even for nibble packing, got {k}")

    grouped = weight.float().reshape(n, k // group_size, group_size)
    group_min = grouped.amin(dim=-1).clamp_max(0)
    group_max = grouped.amax(dim=-1).clamp_min(0)
    scale = (
        ((group_max - group_min) / 15.0).to(scale_dtype).clamp_min(torch.finfo(scale_dtype).tiny)
    )
    scale_f = scale.float()
    zero = torch.round(-group_min / scale_f).clamp(0, 15).to(torch.uint8)
    quantized = (
        torch.round(grouped / scale_f.unsqueeze(-1) + zero.float().unsqueeze(-1))
        .clamp(0, 15)
        .to(torch.uint8)
    )

    unsigned = quantized.reshape(n, k)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    dequantized = (
        (quantized.float() - zero.float().unsqueeze(-1)) * scale_f.unsqueeze(-1)
    ).reshape(n, k)
    return packed.contiguous(), scale.contiguous(), zero.contiguous(), dequantized


def repack_w4a16_weight(packed: torch.Tensor) -> torch.Tensor:
    """Reorder a row-major ``[N, K/2]`` uint8 weight for W4A16 GEMM.

    Args:
        packed: Row-major packed weights, ``[N, K/2]``, ``torch.uint8``.

    Returns:
        A contiguous tensor with the same shape and dtype in the prepacked layout.
    """
    n, kp = packed.shape
    step = 64
    lanes = 4
    if kp % step:
        raise ValueError(f"K/2={kp} must be a multiple of {step}")
    n_steps = kp // step
    words_per_lane = step // (4 * lanes)
    grouped = packed.view(n, n_steps, step).to(torch.int32)
    low, high = grouped & 0xF, (grouped >> 4) & 0xF
    out = torch.zeros(n, n_steps, step // 4, dtype=torch.int32, device=packed.device)
    for lane in range(lanes):
        for word in range(words_per_lane):
            acc = torch.zeros(n, n_steps, dtype=torch.int32, device=packed.device)
            for pair in range(4):
                src = 16 * word + 4 * pair + lane
                acc = acc | (low[:, :, src] << (4 * pair)) | (high[:, :, src] << (4 * pair + 16))
            out[:, :, lane * words_per_lane + word] = acc
    return out.reshape(n, kp // 4).view(torch.uint8).reshape(n, kp).contiguous()


def unrepack_w4a16_weight(prepacked: torch.Tensor) -> torch.Tensor:
    """Invert :func:`repack_w4a16_weight`: the row-major ``[N, K/2]`` packing of a prepacked weight.

    Args:
        prepacked: Prepacked weights, ``[N, K/2]``, ``torch.uint8``.

    Returns:
        The row-major packed weights, two INT4 per byte, even K in the low nibble.
    """
    n, kp = prepacked.shape
    step = 64
    lanes = 4
    n_steps = kp // step
    words_per_lane = step // (4 * lanes)
    words = prepacked.contiguous().view(torch.int32).reshape(n, n_steps, step // 4)
    low = torch.zeros(n, n_steps, step, dtype=torch.int32, device=prepacked.device)
    high = torch.zeros_like(low)
    for lane in range(lanes):
        for word in range(words_per_lane):
            acc = words[:, :, lane * words_per_lane + word]
            for pair in range(4):
                src = 16 * word + 4 * pair + lane
                low[:, :, src] = (acc >> (4 * pair)) & 0xF
                high[:, :, src] = (acc >> (4 * pair + 16)) & 0xF
    return (low | (high << 4)).to(torch.uint8).reshape(n, kp)


def dequantize_w4a16_weight(
    prepacked: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
) -> torch.Tensor:
    """The logical ``[N, K]`` weight a prepacked INT4 weight and its group metadata encode.

    Weight ``(n, k)`` is ``(q - zero[n, g]) * scale[n, g]`` in float32, ``g`` the group of
    ``k`` and ``q`` its INT4 value.
    """
    packed = unrepack_w4a16_weight(prepacked)
    n, kp = packed.shape
    q = torch.empty(n, 2 * kp, dtype=torch.float32, device=packed.device)
    q[:, 0::2] = (packed & 0xF).float()
    q[:, 1::2] = (packed >> 4).float()
    group_size = 2 * kp // scale.shape[1]
    zero_f = zero.float().repeat_interleave(group_size, dim=1)
    scale_f = scale.float().repeat_interleave(group_size, dim=1)
    return (q - zero_f) * scale_f


class GemmW4A16Workload(WorkloadBase):
    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        group_size: int = W4A16_GROUP_SIZE,
    ) -> None:
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.group_size = group_size
        self._row_major_weight: torch.Tensor | None = None

    @classmethod
    def from_call(cls, call: Any) -> "GemmW4A16Workload":
        """The workload of one manifest call of ``GemmW4A16FwdOp``."""
        ix = call.ix
        return cls(ix["M"], ix["N"], ix["K"], getattr(torch, ix["T"]), ix["group_size"])

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return activation, prepacked weight, scale, and zero point."""
        activation = torch.randn(self.m, self.k, device=run_device(), dtype=self.dtype)
        source_weight = torch.randn(self.n, self.k, device=run_device(), dtype=torch.float32) * 0.25
        packed, scale, zero, _ = quantize_weight_int4(
            source_weight, group_size=self.group_size, scale_dtype=self.dtype
        )
        self._row_major_weight = packed
        return activation, repack_w4a16_weight(packed), scale, zero

    @property
    def row_major_weight(self) -> torch.Tensor:
        """The same weight before the repack, which a baseline reorders its own way."""
        if self._row_major_weight is None:
            raise RuntimeError("row_major_weight is available after gen_inputs()")
        return self._row_major_weight

    def ref_program(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        weight = dequantize_w4a16_weight(packed_weight, weight_scale, weight_zero)
        return torch.matmul(activation, weight.to(activation.dtype).T)


class BmmWorkload(WorkloadBase):
    """Workload for batched matmul: a=[B,M,K], b=[B,K,N] -> d=[B,M,N]."""

    def __init__(self, batch: int, m: int, n: int, k: int, dtype: torch.dtype):
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype

    @classmethod
    def from_call(cls, call: Any) -> "BmmWorkload":
        """The workload of one manifest call of ``BmmFwdOp``."""
        ix = call.ix
        return cls(ix["B"], ix["M"], ix["N"], ix["K"], getattr(torch, ix["T"]))

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.randn(self.batch, self.m, self.k, device=run_device(), dtype=self.dtype)
        b = torch.randn(self.batch, self.k, self.n, device=run_device(), dtype=self.dtype)
        return a, b

    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a, b)


class BmmFp8Workload(WorkloadBase):
    """Workload for batched FP8 GEMM.

    ``a`` is ``[B, M, K]``; ``b`` is a contiguous ``[B, K, N]``, or ``[B, N, K]`` under
    ``trans_b``; ``scale_a`` / ``scale_b`` are rank-0 fp32 scalars in ``[0.5, 1.5)``.
    """

    def __init__(
        self,
        batch: int,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        out_dtype: torch.dtype = torch.bfloat16,
        trans_b: bool = False,
    ) -> None:
        self.batch = batch
        self.m = m
        self.n = n
        self.k = k
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.trans_b = trans_b

    @classmethod
    def from_call(cls, call: Any) -> "BmmFp8Workload":
        """The workload of one manifest call of ``BmmFp8FwdOp``."""
        ix = call.ix
        return cls(
            ix["B"],
            ix["M"],
            ix["N"],
            ix["K"],
            getattr(torch, ix["T"]),
            out_dtype=getattr(torch, ix["out_dtype"]),
            trans_b=ix["trans_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        a = (
            (torch.randn(self.batch, self.m, self.k, device=run_device()) * _FP8_INIT_SCALE)
            .to(self.dtype)
            .contiguous()
        )
        b = (
            (torch.randn(self.batch, self.k, self.n, device=run_device()) * _FP8_INIT_SCALE)
            .to(self.dtype)
            .contiguous()
        )
        if self.trans_b:
            b = b.transpose(-2, -1).contiguous()
        scale_a = (0.5 + torch.rand((), device=run_device(), dtype=torch.float32)).contiguous()
        scale_b = (0.5 + torch.rand((), device=run_device(), dtype=torch.float32)).contiguous()
        return a, b, scale_a, scale_b

    def ref_program(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs
        if self.trans_b:
            b = b.transpose(-2, -1)
        a_f = a.float() * scale_a
        b_f = b.float() * scale_b
        out = torch.bmm(a_f, b_f)
        return out.to(self.out_dtype)


def _generate_batch_sizes(batch_sum: int, batch_count: int):
    base_size = batch_sum // batch_count
    remainder = batch_sum % batch_count
    batch_sizes = [base_size] * batch_count
    for i in range(remainder):
        batch_sizes[i] += 1
    return batch_sizes


def _generate_offsets(batch_sizes_list):
    batch_offsets_list = [0]
    for size in batch_sizes_list[:-1]:
        batch_offsets_list.append(batch_offsets_list[-1] + size)
    return batch_offsets_list


class GroupedGemmWorkload(WorkloadBase):
    def __init__(
        self,
        batch_sum: int,
        batch_count: int,
        N: int,
        K: int,
        dtype: torch.dtype,
        transpose_a: bool,
        transpose_b: bool,
    ):
        self.batch_sum = batch_sum
        self.batch_count = batch_count
        self.N = N
        self.K = K
        self.dtype = dtype
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b
        self.batch_sizes_list = _generate_batch_sizes(batch_sum, batch_count)

    @classmethod
    def from_call(cls, call: Any) -> "GroupedGemmWorkload":
        """The workload of one manifest call of ``GroupedGemmFwdOp``."""
        ix = call.ix
        return cls(
            ix["M"],
            ix["G"],
            ix["N"],
            ix["K"],
            getattr(torch, ix["T"]),
            ix["transpose_a"],
            ix["transpose_b"],
        )

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        batch_sizes_list = self.batch_sizes_list
        N, K = self.N, self.K
        device = run_device()
        dtype = self.dtype
        batch_sum = sum(batch_sizes_list)
        batch_count = len(batch_sizes_list)
        batch_offsets_list = _generate_offsets(batch_sizes_list)

        if not self.transpose_a:
            # NT / NN: A is (batch_sum, K)
            A = torch.randn(batch_sum, K, device=device, dtype=dtype)
            if self.transpose_b:
                # NT: B is (batch_count, N, K)
                B = torch.randn(batch_count, N, K, device=device, dtype=dtype)
            else:
                # NN: B is (batch_count, K, N)
                B = torch.randn(batch_count, K, N, device=device, dtype=dtype)
        else:
            # TN / TT: A is (batch_sum, N)
            A = torch.randn(batch_sum, N, device=device, dtype=dtype)
            if self.transpose_b:
                # TT: B is (K, batch_sum)
                B = torch.randn(K, batch_sum, device=device, dtype=dtype)
            else:
                # TN: B is (batch_sum, K)
                B = torch.randn(batch_sum, K, device=device, dtype=dtype)

        batch_sizes = torch.tensor(batch_sizes_list, device=device, dtype=torch.int32)
        batch_offsets = torch.tensor(batch_offsets_list, device=device, dtype=torch.int32)
        return A, B, batch_sizes, batch_offsets

    def ref_program(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        batch_sizes: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.transpose_a:
            # NT / NN: output is (batch_sum, N)
            if self.transpose_b:
                # NT: A @ B^T
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[1]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i].transpose(0, 1).contiguous())
                    start = end
            else:
                # NN: A @ B
                assert A.shape[0] == sum(batch_sizes)
                assert B.shape[0] == len(batch_sizes)
                output = torch.empty((sum(batch_sizes), B.shape[2]), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[start:end] = torch.mm(A[start:end], B[i])
                    start = end
        else:
            # TN / TT: output is (batch_count, N, K)
            total_batch = int(batch_sizes.sum().item())
            assert A.shape[0] == total_batch
            N = A.shape[1]
            batch_count = len(batch_sizes)

            if self.transpose_b:
                # TT: A^T @ B^T
                K = B.shape[0]
                assert B.shape[1] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(
                        A[start:end].transpose(0, 1), B[:, start:end].transpose(0, 1)
                    )
                    start = end
            else:
                # TN: A^T @ B
                K = B.shape[1]
                assert B.shape[0] == total_batch
                output = torch.zeros((batch_count, N, K), device=A.device, dtype=A.dtype)
                start = 0
                for i, size in enumerate(batch_sizes):
                    size = int(size.item())
                    end = start + size
                    output[i] = torch.mm(A[start:end].transpose(0, 1), B[start:end])
                    start = end
        return output
