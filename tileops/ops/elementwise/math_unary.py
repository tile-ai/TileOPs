"""Unary math elementwise ops (exp/log/sqrt/abs/neg/round/etc.)."""


import torch

from tileops.kernels.elementwise import (
    AbsFwdKernel,
    CeilFwdKernel,
    CosFwdKernel,
    ErfFwdKernel,
    ExpFwdKernel,
    Expm1FwdKernel,
    FloorFwdKernel,
    Log1pFwdKernel,
    LogFwdKernel,
    NegFwdKernel,
    ReciprocalFwdKernel,
    RoundFwdKernel,
    RsqrtFwdKernel,
    SignFwdKernel,
    SinFwdKernel,
    SqrtFwdKernel,
    TruncFwdKernel,
)

from ._base import (
    _MANIFEST_INT_DTYPES,
    KernelEntry,
    UnaryOp,
    _IntIdentityUnaryOp,
    resolve_output_dtype,
)


class ExpFwdOp(UnaryOp):
    """Element-wise exp(x)."""

    _op_name = "exp"
    kernel_cls = ExpFwdKernel


class LogFwdOp(UnaryOp):
    """Element-wise log(x)."""

    _op_name = "log"
    kernel_cls = LogFwdKernel


class SqrtFwdOp(UnaryOp):
    """Element-wise sqrt(x)."""

    _op_name = "sqrt"
    kernel_cls = SqrtFwdKernel


class RsqrtFwdOp(UnaryOp):
    """Element-wise 1/sqrt(x)."""

    _op_name = "rsqrt"
    kernel_cls = RsqrtFwdKernel


class AbsFwdOp(_IntIdentityUnaryOp):
    """Element-wise |x|."""

    _op_name = "abs"
    kernel_cls = AbsFwdKernel
    _int_handler = staticmethod(torch.abs)


class NegFwdOp(_IntIdentityUnaryOp):
    """Element-wise -x."""

    _op_name = "neg"
    kernel_cls = NegFwdKernel
    _int_handler = staticmethod(torch.neg)


class ReciprocalFwdOp(UnaryOp):
    """Element-wise 1/x.

    Mirrors ``torch.reciprocal`` int-input promotion: integral dtypes
    (uint8 / int8 / int16 / int32 / int64) are cast to float32 before the
    float kernel runs, so their entry carries ``compute_dtype`` and
    ``output_dtype`` of float32 against an integral key. Floating inputs
    (float16 / bfloat16 / float32) follow the standard same-dtype path.
    """

    _op_name = "reciprocal"
    kernel_cls = ReciprocalFwdKernel

    def _build_entry(self, dtype: torch.dtype) -> KernelEntry:
        """An integer input computes in float32; the entry records both types.

        The semantic dtype keys the entry and drives roofline accounting —
        integer input bytes, float32 output bytes — while the kernel is built
        for the compute dtype the float-only kernel requires.
        """
        compute = torch.float32 if dtype in _MANIFEST_INT_DTYPES else dtype
        return KernelEntry(
            kernel=self._build_kernel_instance(
                N_total=self.N_total, dtype=compute, tune=self.tune,
            ),
            compute_dtype=compute,
            output_dtype=resolve_output_dtype(type(self).__name__, dtype),
        )

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Promote here, past the boundary, so the caller's dtype stays visible."""
        entry = self._entry(input.dtype)
        flat = input.contiguous().reshape(-1)
        if entry.compute_dtype != input.dtype:
            flat = flat.to(entry.compute_dtype)
        return entry.kernel(flat).reshape(input.shape)


class SignFwdOp(_IntIdentityUnaryOp):
    """Element-wise sign(x): -1, 0, or +1."""

    _op_name = "sign"
    kernel_cls = SignFwdKernel
    # Manifest: flops = "2 * N" (two compares + selects per element).
    FLOPS_PER_ELEM = 2
    _int_handler = staticmethod(torch.sign)


class SinFwdOp(UnaryOp):
    """Element-wise sin(x)."""

    _op_name = "sin"
    kernel_cls = SinFwdKernel


class CosFwdOp(UnaryOp):
    """Element-wise cos(x)."""

    _op_name = "cos"
    kernel_cls = CosFwdKernel


class FloorFwdOp(_IntIdentityUnaryOp):
    """Element-wise floor(x)."""

    _op_name = "floor"
    kernel_cls = FloorFwdKernel


class CeilFwdOp(_IntIdentityUnaryOp):
    """Element-wise ceil(x)."""

    _op_name = "ceil"
    kernel_cls = CeilFwdKernel


class RoundFwdOp(_IntIdentityUnaryOp):
    """Element-wise round(x) to ``decimals`` decimal places.

    The underlying kernel performs banker's round-to-nearest-integer, matching
    ``torch.round`` for ``decimals=0``. Non-zero ``decimals`` is supported at
    the op layer via the standard decomposition:
    ``round(x, decimals=k) == round(x * 10**k) / 10**k``.

    Args:
        N_total: Total number of elements (flattened).
        kernel_map: Optional kernel dispatch override.
        tune: Whether to autotune.
    """

    _op_name = "round"
    kernel_cls = RoundFwdKernel

    def forward(
        self, input: torch.Tensor, decimals: int = 0,
    ) -> torch.Tensor:
        if decimals == 0:
            return super().forward(input)
        # Non-zero decimals path still owes the same input contract as the
        # ``decimals=0`` fast path (UnaryOp.forward). Run the shared validator
        # before any fp32 arithmetic so a CPU tensor / wrong dtype / wrong
        # numel cannot silently bypass the checks.
        self._validate_input(input)
        # Integer dtypes are no-ops regardless of decimals (rounding an int
        # produces the same int). Match the float-path identity contract.
        if input.dtype in _MANIFEST_INT_DTYPES:
            return input.clone()
        # Run through fp32 so low-precision inputs (fp16/bf16) cannot overflow
        # when ``torch.round`` internally scales by ``10**decimals`` — e.g.
        # ``100 * 10**4 = 1e6`` exceeds fp16 max (~65504). The single down-cast
        # at the end restores the op's contract dtype. The manifest's
        # ``kernel_map`` continues to describe the round-to-nearest-integer
        # kernel that handles the ``decimals=0`` fast path above.
        # Metadata only; the cast target comes from the tensor.
        return torch.round(input.float(), decimals=decimals).to(input.dtype)


class TruncFwdOp(_IntIdentityUnaryOp):
    """Element-wise trunc(x)."""

    _op_name = "trunc"
    kernel_cls = TruncFwdKernel


class ErfFwdOp(UnaryOp):
    """Element-wise erf(x)."""

    _op_name = "erf"
    kernel_cls = ErfFwdKernel


class Log1pFwdOp(UnaryOp):
    """Element-wise log(1 + x)."""

    _op_name = "log1p"
    kernel_cls = Log1pFwdKernel
    # Manifest: flops = "2 * N" (1 add + 1 log).
    FLOPS_PER_ELEM = 2


class Expm1FwdOp(UnaryOp):
    """Element-wise exp(x) - 1."""

    _op_name = "expm1"
    kernel_cls = Expm1FwdKernel
    # Manifest: flops = "2 * N" (1 exp + 1 sub).
    FLOPS_PER_ELEM = 2
