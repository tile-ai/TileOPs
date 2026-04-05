import torch

from workloads.base import WorkloadBase

CONV_KERNEL_SIZE = 4


class EngramGateConvFwdTest(WorkloadBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self):
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda")
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device="cuda") * 0.02
        return H, k, v, rms_w_h, rms_w_v, conv_w
