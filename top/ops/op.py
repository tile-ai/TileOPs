import torch
from tilelang.profiler import do_bench
from top.kernels.kernel import Kernel
from top.utils import get_arch
from typing import Optional, Union, Dict
from abc import abstractmethod, ABC


class Op(ABC):
    """Base class for TileOPs operations.
    
    A Op represents a computational operation with:
    - Hardware-aware kernel dispatch
    - Correctness testing via reference implementation
    - Performance profiling
    - Autotuning interface
    
    Examples:
        >>> from top import mha_fwd  # mha_fwd is a subclass of Op
        >>> op = mha_fwd(batch=1, heads=8, seq_len=512, dim=64, is_causal=True)
        >>> Q, K, V = op.gen_inputs()
        >>> output = op(Q, K, V)
        >>> op.check()  # Verify correctness
        >>> latency = op.profile()  # Benchmark performance
    
    Attributes:
        kernel: top.Kernel instance (e.g. mha_fwd_kernel)
        dtype: Data type for computation (e.g., torch.float16)
        device: Device for computation (e.g., 'cuda')
        input_shapes: Expected input tensor shapes

    Properties:
        total_flops (optional): Total flops for the op.
            If specified, will be used to calculate TFlops in profile().
        total_memory (optional): Total memory for the op.
            If specified, will be used to calculate Bandwidth in profile().
    """

    kernel: Kernel
    kernel_map: Optional[Dict[str, Kernel]] = {}
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None

    @property
    @abstractmethod
    def default_kernel_map(self) -> Dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def dispatch_kernel(self, kernel_map: Optional[Dict[str, Kernel]] = None):
        assert self.default_kernel_map is not None and len(self.default_kernel_map) > 0
        for name, default_kernel in self.default_kernel_map.items():
            if kernel_map is not None and name in kernel_map:
                kernel_type = kernel_map[name]
            else:
                kernel_type = default_kernel
            current_arch = get_arch()
            assert current_arch in kernel_type.supported_archs, \
                f'{kernel_type.__name__} is not supported on architecture {current_arch}'
            self.kernel_map[name] = kernel_type

    @property
    def total_flops(self):
        return None

    @property
    def total_memory(self):
        return None

    def autotune(self):
        """Autotune all kernels of the op"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args, **kwargs):
        """Make the op callable - delegates to forward()"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def ref_program(self, *inputs):
        raise NotImplementedError("ref_program method is not implemented")

    def gen_inputs(self):
        """Generate random inputs for the op"""
        assert self.input_shapes is not None, "input_shapes is not set for default gen_inputs()"
        return tuple(
            torch.randn(shape, device=self.device, dtype=self.dtype) for shape in self.input_shapes)
        # NOTE: by default all dtypes are self.dtype, should override this method in subclasses if not

    def check(self, atol=1e-2, rtol=1e-2):
        """Check the correctness of the op"""
        inputs = self.gen_inputs()

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
            outputs = self.forward(*inputs)

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

        print(f"All checks passed for {self.__class__.__name__}.✅")

    def profile(self, warmup=25, rep=100):
        """Profile the op, and print relevant metrics"""
        print(f"===== Profiling {self.__class__.__name__} =====")
        inputs = self.gen_inputs()
        with torch.no_grad():
            latency = do_bench(
                lambda: self.forward(*inputs), warmup=warmup, rep=rep,
                backend='cupti')  # Always use cupti backend for better accuracy

        print(f"{self.__class__.__name__} latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(
                f"{self.__class__.__name__} TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops")
        if self.total_memory is not None:
            print(
                f"{self.__class__.__name__} Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s"
            )
