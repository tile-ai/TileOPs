import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase

CONV_KERNEL_SIZE = 4


def _ref_rmsnorm(x, w, eps=1e-6):
    """Returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


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

    def ref_program(self, H, k, v, rms_w_h, rms_w_v, conv_w):
        return ref_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, self.eps)


def ref_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """PyTorch reference for Engram GateConv forward."""
    M, T, d = H.shape

    h_norm, rrms_h = _ref_rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _ref_rmsnorm(k, rms_w_h, eps)  # same weight

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))  # (M, T, 1)

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _ref_rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    # Causal DWConv1D
    v_perm = v_hat_norm.float().permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    conv_w_expanded = conv_w.float().T.unsqueeze(1)  # (d, 1, ks)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (
        Y.to(H.dtype),
        v_hat.to(H.dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )
