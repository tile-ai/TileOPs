import torch
from abc import ABC, abstractmethod
from tilelang.profiler import do_bench
from top.ops import Op


class Benchmark(ABC):

    op_type: type[Op]

    @property
    def total_flops(self):
        raise NotImplementedError

    @property
    def total_memory(self):
        raise NotImplementedError

    def gen_inputs(self):
        raise NotImplementedError
        #TODo: impl this?

    @abstractmethod
    def ref_program(self, *inputs):
        raise NotImplementedError

    def check(self, op, *inputs, atol=1e-2, rtol=1e-2):
        """Check the correctness of the op"""

        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            else:
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
            # print(f"outputs[{i}] max err: {(output - output_ref).abs().max()}")
            if output_ref is not None:  # skip checking for None placeholders in ref
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {(output - output_ref).abs().max()}"

        print(f"All checks passed for {op.__class__.__name__}.✅")

    def check_fn(self, fn, *inputs, atol=1e-2, rtol=1e-2, grad=True):
        """Check the correctness of the function and layer"""

        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            else:
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
            for input in inputs:
                outputs.append(input.grad)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(
            outputs_ref
        ), f"outputs: {len(outputs)}  and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            # print(f"outputs[{i}] max err: {(output - output_ref).abs().max()}")
            if output_ref is not None:  # skip checking for None placeholders in ref
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {(output - output_ref).abs().max()}"

        print(f"All checks passed for {fn.__class__.__name__}.✅")

    def profile(self, op, *inputs, warmup=100, rep=100):
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
            print(
                f"{op.__class__.__name__} tl-Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s"
            )

    def baseline_profile(self,
                         baseline_op,
                         *inputs,
                         backend="Base",
                         warmup=100,
                         rep=100,
                         device="cuda:0"):
        """Benchmark the perf of the baselin op"""

        print(f"===== Profiling {backend} =====")
        print(f"{backend} profile with warmup: {warmup}, rep: {rep}")

        # Warmup to get rid of CUDA lazy initialization effects.
        for _ in range(warmup):
            _ = baseline_op(*inputs)
        torch.cuda.synchronize(device=device)

        # CUDA event-based timing for higher precision.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(rep):
            _ = baseline_op(*inputs)
        end_event.record()

        torch.cuda.synchronize(device=device)
        total_ms = start_event.elapsed_time(end_event)
        latency = total_ms / float(rep)

        print(f"{backend} Baseline-latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(f"{backend} Baseline-TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops")
        if self.total_memory is not None:
            print(f"{backend} Baseline-Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s")
