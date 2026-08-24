"""Strategy selection policy for elementwise kernels."""

import warnings
from dataclasses import dataclass

import torch

from ._dtype import _is_fp8
from ._output import _bool_output_needs_scalar


@dataclass(frozen=True)
class UnaryStrategyPolicy:
    """Strategy selection rules for one-input elementwise kernels."""

    strategies: list[str]
    default_strategy: str

    def choose(
        self,
        *,
        requested: str | None,
        input_dtype: torch.dtype,
        declared_output_dtype: torch.dtype | None,
    ) -> str:
        self._validate_requested(requested)
        if input_dtype == torch.bool:
            self._warn_direct_override(
                requested,
                "UnaryKernel: dtype=torch.bool requires strategy="
                "'direct' (TileLang cannot lower vectorised boolx<N> "
                "loads); overriding requested strategy={requested!r}.",
            )
            return "direct"
        if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
            self._warn_direct_override(
                requested,
                "UnaryKernel: dtype={dtype} with torch.bool output "
                "requires strategy='direct' (TileLang cannot lower "
                "vectorised boolx<N> stores for sub-32-bit integer "
                "inputs); overriding requested strategy={requested!r}.",
                dtype=input_dtype,
            )
            return "direct"
        if requested is None and _is_fp8(input_dtype):
            return "explicit_parallel"
        return requested or self.default_strategy

    def _validate_requested(self, requested: str | None) -> None:
        if requested is not None and requested not in self.strategies:
            raise ValueError(f"Unknown strategy '{requested}', expected one of {self.strategies}")

    @staticmethod
    def _warn_direct_override(
        requested: str | None,
        template: str,
        **kwargs,
    ) -> None:
        if requested is None or requested == "direct":
            return
        warnings.warn(
            template.format(requested=requested, **kwargs),
            RuntimeWarning,
            stacklevel=3,
        )


@dataclass(frozen=True)
class BinaryStrategyPolicy:
    """Strategy selection rules for broadcast-capable binary elementwise kernels."""

    strategies: list[str]
    default_strategy: str

    def choose(
        self,
        *,
        requested: str | None,
        input_dtype: torch.dtype,
        declared_output_dtype: torch.dtype | None,
        same_shape: bool,
    ) -> str:
        self._validate_requested(requested)
        if input_dtype == torch.bool:
            self._warn_direct_override(
                requested,
                "BinaryKernel: dtype=torch.bool requires strategy="
                "'direct' (TileLang cannot lower vectorised boolx<N> "
                "loads); overriding requested strategy={requested!r}.",
            )
            return "direct"
        if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
            self._warn_direct_override(
                requested,
                "BinaryKernel: dtype={dtype} with torch.bool output "
                "requires strategy='direct' (TileLang cannot lower "
                "vectorised boolx<N> stores for sub-32-bit integer "
                "inputs); overriding requested strategy={requested!r}.",
                dtype=input_dtype,
            )
            return "direct"
        if requested is not None:
            if requested == "register_copy" and not same_shape:
                return "explicit_parallel"
            return requested
        return "register_copy" if same_shape else self.default_strategy

    def _validate_requested(self, requested: str | None) -> None:
        if requested is not None and requested not in self.strategies:
            raise ValueError(f"Unknown strategy '{requested}', expected one of {self.strategies}")

    @staticmethod
    def _warn_direct_override(
        requested: str | None,
        template: str,
        **kwargs,
    ) -> None:
        if requested is None or requested == "direct":
            return
        warnings.warn(
            template.format(requested=requested, **kwargs),
            RuntimeWarning,
            stacklevel=3,
        )
