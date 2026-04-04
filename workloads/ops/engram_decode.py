import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


def _ref_rmsnorm(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


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

    def ref_program(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        y_ref, state_ref = _ref_engram_decode_step(
            e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
            self.max_conv_len, self.dilation, self.eps,
        )
        return y_ref, state_ref


def _ref_engram_decode_step(
    e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
    max_conv_len, dilation, eps=1e-6,
):
    """PyTorch reference for a single decode step with dilated causal conv.

    Args:
        e_t:        (B, d_mem)
        h_t:        (B, d)
        conv_state: (B, L, d) — cached v_hat_norm history (L <= max_conv_len)
        W_K, W_V:   (d_mem, d)
        rms_w_h, rms_w_v: (d,)
        conv_w:     (w, d) — conv kernel weights (w = kernel size, model param)
        max_conv_len: max cache capacity
        dilation:   dilation factor δ
    Returns:
        y_t:            (B, d)
        new_conv_state: (B, L', d) where L' = min(L+1, max_conv_len)
    """
    B, d = h_t.shape
    w = conv_w.shape[0]  # conv kernel size
    L = conv_state.shape[1]  # current history length

    # Projection
    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    # RMSNorm gating
    h_norm, _ = _ref_rmsnorm(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _ref_rmsnorm(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v

    # RMSNorm(v_hat)
    v_hat_norm, _ = _ref_rmsnorm(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)  # (B, d)

    # Dilated causal conv
    # Left-pad state to max_conv_len, then read dilated positions
    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    # Window: w values spaced by dilation
    # tap p: state[max_conv_len - (w-1-p)*dilation]  for p=0..w-2
    # tap w-1: current v_hat_norm
    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, state_idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    # Update state: shift left by 1, append v_hat_norm
    if max_conv_len > L:
        new_conv_state = torch.cat([
            conv_state,
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)
    else:
        new_conv_state = torch.cat([
            conv_state[:, 1:, :],
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ], dim=1)

    # SiLU + residual
    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state
