"""Launch configuration policy for elementwise kernels."""

from dataclasses import dataclass

import torch

from ._dtype import _is_fp8, _torch_dtype_nbytes


@dataclass(frozen=True)
class ElementwiseLaunchConfig:
    """Concrete launch knobs for one elementwise specialization."""

    strategy: str
    threads: int
    num_per_thread: int

    def as_dict(self) -> dict:
        """Return the config shape expected by ``Kernel.init_config``."""
        return {
            "strategy": self.strategy,
            "threads": self.threads,
            "num_per_thread": self.num_per_thread,
        }


@dataclass(frozen=True)
class ElementwiseLaunchPolicy:
    """Heuristics that turn element dtype and shape into launch knobs."""

    default_threads: int = 128
    bytes_per_thread: int = 16
    min_num_per_thread: int = 4
    bool_output_max_num_per_thread: int = 4
    max_threads: int = 1024
    target_blocks: int = 256
    fp8_num_per_thread: int = 16

    def default_config(
        self,
        *,
        strategy: str,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        n_total: int | None,
        stores_bool: bool = True,
    ) -> ElementwiseLaunchConfig:
        """Return the default launch config for this specialization.

        The rule order is significant: fp8 keeps its historical 128-bit default;
        otherwise start from input width, cap real bool stores, widen for narrow
        non-bool outputs, then shrink small non-direct launches.
        """
        if _is_fp8(input_dtype):
            return ElementwiseLaunchConfig(
                strategy=strategy,
                threads=self.default_threads,
                num_per_thread=self.fp8_num_per_thread,
            )
        threads = self.default_threads
        npt = self._base_num_per_thread(input_dtype)
        threads, npt, handled_bool = self._adjust_for_bool_output(
            threads,
            npt,
            output_dtype,
            stores_bool,
        )
        if not handled_bool:
            npt = self._adjust_for_narrow_output(npt, input_dtype, output_dtype)
        npt = self._shrink_npt_for_small_tensor(strategy, n_total, threads, npt)
        return ElementwiseLaunchConfig(strategy=strategy, threads=threads, num_per_thread=npt)

    def _base_num_per_thread(self, dtype: torch.dtype) -> int:
        elem_bytes = _torch_dtype_nbytes(dtype)
        return max(self.min_num_per_thread, self.bytes_per_thread // elem_bytes)

    def _adjust_for_bool_output(
        self,
        threads: int,
        npt: int,
        output_dtype: torch.dtype,
        stores_bool: bool,
    ) -> tuple[int, int, bool]:
        if output_dtype != torch.bool or not stores_bool:
            return threads, npt, False
        capped = min(npt, self.bool_output_max_num_per_thread)
        return min(self.max_threads, threads * npt // capped), capped, True

    def _adjust_for_narrow_output(
        self,
        npt: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> int:
        elem_bytes = _torch_dtype_nbytes(input_dtype)
        stored_bytes = _torch_dtype_nbytes(output_dtype)
        return npt * 2 if stored_bytes < elem_bytes else npt

    def _shrink_npt_for_small_tensor(
        self,
        strategy: str,
        n_total: int | None,
        threads: int,
        npt: int,
    ) -> int:
        if n_total is None or strategy == "direct":
            return npt
        while npt > self.min_num_per_thread and n_total < threads * npt * self.target_blocks:
            npt //= 2
        return npt


_LAUNCH_POLICY = ElementwiseLaunchPolicy()
_DEFAULT_THREADS: int = _LAUNCH_POLICY.default_threads
