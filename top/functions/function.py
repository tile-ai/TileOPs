from top import Kernel

class Function:

    def autotune(self):
        for attr in dir(self):
            if isinstance(attr, Kernel) and hasattr(attr, 'autotune'):
                attr.autotune()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method is not implemented")

    __call__ = forward

    def ref_program(self, *args, **kwargs):
        raise NotImplementedError("ref_program method is not implemented")

    def gen_inputs(self):
        raise NotImplementedError("gen_inputs method is not implemented")

    def check(self):
        raise NotImplementedError("check method is not implemented")

    def profile(self, *args, **kwargs):
        raise NotImplementedError("profile method is not implemented")