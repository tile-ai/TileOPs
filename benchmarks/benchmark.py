from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import torch
from tilelang.profiler import do_bench

from top.ops import Op


class Benchmark(ABC):

    op_type: type[Op]

    @property
    def total_flops(self) -> Optional[float]:
        raise NotImplementedError

    @property
    def total_memory(self) -> Optional[float]:
        raise NotImplementedError

    def gen_inputs(self) -> Any:
        raise NotImplementedError
        # TODO: impl this?

    @abstractmethod
    def ref_program(self, *inputs: Tuple[torch.Tensor]) -> Any:  # noqa: U100
        raise NotImplementedError

    def check(self,
              op: Op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check the correctness of the op"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            if output_ref is not None:  # skip checking for None placeholders in ref
                max_err = (output - output_ref).abs().max()
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {max_err}"

        print(f"All checks passed for {op.__class__.__name__}.✅")

    def check_fn(self,
                 fn: callable,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 grad: bool = True) -> None:
        """Check the correctness of the function and layer"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        if not grad:
            with torch.no_grad():
                outputs = fn(*inputs)
        else:
            output = fn(*inputs)
            loss = output.sum()
            loss.backward()
            outputs = []
            outputs.append(output)
            for inp in inputs:
                outputs.append(inp.grad)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            if output_ref is not None:  # skip checking for None placeholders in ref
                max_err = (output - output_ref).abs().max()
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {max_err}"

        print(f"All checks passed for {fn.__class__.__name__}.✅")

    def profile(self,
                op: Op,
                *inputs: Tuple[torch.Tensor],
                warmup: int = 100,
                rep: int = 100) -> None:
        """Benchmark the perf of the op"""
        print(f"===== Profiling {op.__class__.__name__} =====")
        print(f"{op.__class__.__name__} profile with warmup: {warmup}, rep: {rep}")
        with torch.no_grad():
            # Always use cupti backend for better accuracy
            latency = do_bench(lambda: op(*inputs), warmup=warmup, rep=rep, backend='cupti')

        print(f"{op.__class__.__name__} tl-latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(
                f"{op.__class__.__name__} tl-TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops"
            )
        if self.total_memory is not None:
            bandwidth = self.total_memory / latency * 1e-9
            print(f"{op.__class__.__name__} tl-Bandwidth: {bandwidth:.2f} GB/s")

    def baseline_profile(self,
                         baseline_op: Op,
                         *inputs: Tuple[torch.Tensor],
                         backend: str = "Base",
                         warmup: int = 100,
                         rep: int = 100,
                         device: str = "cuda:0") -> None:  # noqa: U100
        """Benchmark the perf of the baselin op"""
        print(f"===== Profiling {backend} =====")
        print(f"{backend} profile with warmup: {warmup}, rep: {rep}")

        with torch.no_grad():
            # Always use cupti backend for better accuracy
            latency = do_bench(
                lambda: baseline_op(*inputs), warmup=warmup, rep=rep, backend='cupti')

        print(f"{backend} Baseline-latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(f"{backend} Baseline-TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops")
        if self.total_memory is not None:
            print(f"{backend} Baseline-Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s")
