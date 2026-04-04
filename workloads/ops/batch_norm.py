import torch

from workloads.base import WorkloadBase


def _make_tensors(N, C, spatial, dtype, device="cuda"):
    shape = (N, C, *spatial)
    x = torch.randn(*shape, device=device, dtype=dtype)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)
    return x, weight, bias, running_mean, running_var


def _ref_fwd(x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """Reference: torch.nn.functional.batch_norm (float32 upcast)."""
    x32 = x.float()
    rm = running_mean.clone()
    rv = running_var.clone()
    y32 = torch.nn.functional.batch_norm(
        x32, rm, rv, weight.float(), bias.float(),
        training=training, momentum=momentum, eps=eps)
    return y32.to(x.dtype), rm, rv


class BatchNormBwdTest(WorkloadBase):

    def __init__(self, N, C, spatial, dtype):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype

    def gen_inputs(self):
        x, weight, bias, running_mean, running_var = _make_tensors(
            self.N, self.C, self.spatial, self.dtype)
        grad_out = torch.randn_like(x)
        # Need mean/rstd from a forward pass.
        x32 = x.float()
        # Compute mean and rstd via native batch norm internals.
        C = self.C
        L = x32.numel() // C
        x_cl = x32.permute(1, 0, *range(2, x32.ndim)).reshape(C, L).contiguous()
        mean = x_cl.mean(dim=1)
        var = x_cl.var(dim=1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
        return grad_out, x, weight, mean, rstd

    def ref_program(self, grad_out, x, weight, mean, rstd):
        """Reference via torch.autograd on a float32 graph."""
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(self.C, device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(self.C, device=x.device, dtype=torch.float32)
        rv = torch.ones(self.C, device=x.device, dtype=torch.float32)
        y32 = torch.nn.functional.batch_norm(
            x32, rm, rv, w32, b32, training=True, momentum=0.1, eps=1e-5)
        y32.backward(grad_out.float())
        return x32.grad.to(x.dtype), w32.grad, b32.grad


class BatchNormFwdTest(WorkloadBase):

    def __init__(self, N, C, spatial, dtype, training):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.training = training

    def gen_inputs(self):
        return _make_tensors(self.N, self.C, self.spatial, self.dtype)

    def ref_program(self, x, weight, bias, running_mean, running_var):
        y, rm, rv = _ref_fwd(x, weight, bias, running_mean, running_var,
                             training=self.training)
        return (y,)
