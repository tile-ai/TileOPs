import torch

from workloads.base import WorkloadBase


class EngramDecodeTest(WorkloadBase):
    def __init__(self, batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, eps=1e-6):
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self):
        e_t = torch.randn(self.batch, self.d_mem, dtype=self.dtype, device="cuda") * 0.1
        h_t = torch.randn(self.batch, self.d, dtype=self.dtype, device="cuda")
        # Full conv_state (max_conv_len entries)
        conv_state = torch.randn(
            self.batch, self.max_conv_len, self.d,
            dtype=self.dtype, device="cuda"
        ) * 0.1
        W_K = torch.randn(self.d_mem, self.d, dtype=self.dtype, device="cuda") * 0.02
        W_V = torch.randn(self.d_mem, self.d, dtype=self.dtype, device="cuda") * 0.02
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(self.conv_kernel_size, self.d, dtype=self.dtype, device="cuda") * 0.02
        return e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w
