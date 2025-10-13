import torch
from tilelang import JITKernel
from tilelang.profiler import do_bench
from top import Kernel
from typing import Optional, Union

class Function:
    kernel: JITKernel = None
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None
    total_flops: Optional[float] = None
    total_memory: Optional[float] = None

    def autotune(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    __call__ = forward

    def ref_program(self, *inputs):
        raise NotImplementedError("ref_program method is not implemented")

    def gen_inputs(self):
        assert self.input_shapes is not None, "input_shapes is not set for default gen_inputs()"
        return tuple(torch.randn(shape, device=self.device, dtype=self.dtype) for shape in self.input_shapes)
        # NOTE: by default all dtypes are self.dtype, should override this method in subclasses if not

    def check(self, atol=1e-2, rtol=1e-2):
        inputs = self.gen_inputs()
        outputs = self.forward(*inputs)
        outputs_ref = self.ref_program(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for output, output_ref in zip(outputs, outputs_ref):
            if output_ref is not None:  # skip checking for None placeholders in ref
                torch.testing.assert_close(output, output_ref, atol=atol, rtol=rtol)
        print(f"All checks passed for {self.__class__.__name__}.✅")
            
    def profile(self, warmup=25, rep=100):
        #TODO: add cupti backend for better accuracy
        print(f"===== Profiling {self.__class__.__name__} =====")
        inputs = self.gen_inputs()
        with torch.no_grad():
            latency = do_bench(lambda: self.forward(*inputs), warmup=warmup, rep=rep)
        print(f"{self.__class__.__name__} latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(f"{self.__class__.__name__} TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops")
        if self.total_memory is not None:
            print(f"{self.__class__.__name__} Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s")