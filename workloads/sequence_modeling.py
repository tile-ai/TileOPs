"""Workloads for the sequence-modeling ops: Engram and MHC."""

import math

import torch
import torch.nn.functional as F

from workloads.workload_base import WorkloadBase

CONV_KERNEL_SIZE = 4


class EngramGateConvFwdWorkload(WorkloadBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda")
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device="cuda") * 0.02
        return H, k, v, rms_w_h, rms_w_v, conv_w

    def ref_program(self, H, k, v, rms_w_h, rms_w_v, conv_w):
        return engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, self.eps)


class EngramGateConvBwdWorkload(WorkloadBase):
    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        """Generate inputs including saved intermediates from a reference forward."""
        H = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda")
        k = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        v = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(CONV_KERNEL_SIZE, self.d, dtype=self.dtype, device="cuda") * 0.02
        dY = torch.randn(self.M, self.seq_len, self.d, dtype=self.dtype, device="cuda") * 0.1

        # Compute saved intermediates via reference forward
        def _rmsnorm(x, w):
            x_f = x.float()
            rrms = (x_f**2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x_f * rrms * w.float(), rrms.squeeze(-1)

        h_norm, rrms_h = _rmsnorm(H, rms_w_h)
        k_norm, rrms_k = _rmsnorm(k, rms_w_h)
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid(dot / (self.d**0.5))
        v_hat = alpha * v.float()
        _, rrms_v = _rmsnorm(v_hat.to(self.dtype), rms_w_v)

        vhat = v_hat.to(self.dtype)
        alpha_squeezed = alpha.squeeze(-1).float()
        rrms_h = rrms_h.float()
        rrms_k = rrms_k.float()
        rrms_v = rrms_v.float()

        return (dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha_squeezed, rrms_h, rrms_k, rrms_v)

    def ref_program(
        self, dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v
    ):
        return ref_engram_gate_conv_bwd(
            dY,
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
            vhat,
            alpha,
            rrms_h,
            rrms_k,
            rrms_v,
            self.eps,
        )


class EngramDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        dtype,
        eps=1e-6,
        conv_len=None,
    ):
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.dtype = dtype
        self.eps = eps
        # The history steps conv_state holds; a full cache by default.
        self.conv_len = max_conv_len if conv_len is None else conv_len

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        e_t = torch.randn(self.batch, self.d_mem, dtype=self.dtype, device="cuda") * 0.1
        h_t = torch.randn(self.batch, self.d, dtype=self.dtype, device="cuda")
        conv_state = (
            torch.randn(self.batch, self.conv_len, self.d, dtype=self.dtype, device="cuda") * 0.1
        )
        W_K = torch.randn(self.d_mem, self.d, dtype=self.dtype, device="cuda") * 0.02
        W_V = torch.randn(self.d_mem, self.d, dtype=self.dtype, device="cuda") * 0.02
        rms_w_h = torch.ones(self.d, dtype=self.dtype, device="cuda")
        rms_w_v = torch.ones(self.d, dtype=self.dtype, device="cuda")
        conv_w = torch.randn(self.conv_kernel_size, self.d, dtype=self.dtype, device="cuda") * 0.02
        return e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w

    def ref_program(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        y_ref, state_ref = engram_decode_step_torch(
            e_t,
            h_t,
            conv_state,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
            self.max_conv_len,
            self.dilation,
            self.eps,
        )
        return y_ref, state_ref


def _rmsnorm(x, w, eps=1e-6):
    """Returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """PyTorch reference for Engram GateConv forward."""
    M, T, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    conv_w_expanded = conv_w.float().T.unsqueeze(1)
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


def ref_engram_gate_conv_bwd(
    dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6
):
    """PyTorch reference backward via autograd."""
    M, T, d = H.shape

    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    def _rmsnorm(x, w):
        return x * (x**2).mean(dim=-1, keepdim=True).add(eps).rsqrt() * w

    h_norm = _rmsnorm(H_ag, w_h_ag)
    k_norm = _rmsnorm(k_ag, w_h_ag)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (d**0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm = _rmsnorm(v_hat_ag, w_v_ag)

    v_perm = v_hat_norm.permute(0, 2, 1)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    cw_expanded = cw_ag.T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, cw_expanded, groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag

    # On this thread, not autograd's engine thread: a benchmark timing this
    # reference cannot attribute kernels launched where its iteration id was
    # never pushed. Same gradients either way.
    with torch.autograd.set_multithreading_enabled(False):
        Y_ag.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(H.dtype),
        v_ag.grad.to(H.dtype),
        w_h_ag.grad,
        w_v_ag.grad,
        cw_ag.grad,
    )


def _rmsnorm_decode(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def engram_decode_step_torch(
    e_t,
    h_t,
    conv_state,
    W_K,
    W_V,
    rms_w_h,
    rms_w_v,
    conv_w,
    max_conv_len,
    dilation,
    eps=1e-6,
):
    """PyTorch reference for a single decode step with dilated causal conv."""
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1).to(h_t.dtype), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1).to(h_t.dtype), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    if max_conv_len > L:
        padded_state = F.pad(conv_state.float(), (0, 0, max_conv_len - L, 0))
    else:
        padded_state = conv_state.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, state_idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    if max_conv_len > L:
        new_conv_state = torch.cat(
            [
                conv_state,
                v_hat_norm.unsqueeze(1).to(conv_state.dtype),
            ],
            dim=1,
        )
    else:
        new_conv_state = torch.cat(
            [
                conv_state[:, 1:, :],
                v_hat_norm.unsqueeze(1).to(conv_state.dtype),
            ],
            dim=1,
        )

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state


class MHCPreWorkload(WorkloadBase):
    """One MHC pre case: its shape, its dtype, and the scalars it is run with.

    The scaling and sinkhorn scalars are manifest ``params``, so they are construction
    arguments of the op; the case holds the values it builds that op with.
    """

    def __init__(
        self,
        batch: int,
        n_expand: int,
        c_x: int,
        dtype: torch.dtype,
        *,
        alpha_pre: float | None = None,
        alpha_post: float | None = None,
        alpha_res: float | None = None,
        sinkhorn_repeat: int = 20,
        sinkhorn_eps: float = 0.02,
    ):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.alpha_pre = float(torch.randn(())) if alpha_pre is None else alpha_pre
        self.alpha_post = float(torch.randn(())) if alpha_post is None else alpha_post
        self.alpha_res = float(torch.randn(())) if alpha_res is None else alpha_res
        self.sinkhorn_repeat = sinkhorn_repeat
        self.sinkhorn_eps = sinkhorn_eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        phi = torch.randn(
            [n_expand * c_x, n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32
        )
        x = torch.randn([batch, n_expand * c_x], device="cuda", dtype=torch.bfloat16)
        b = torch.randn([n_expand * n_expand + 2 * n_expand], device="cuda", dtype=torch.float32)
        return phi, x, b

    def ref_program(
        self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return mhc_pre_ref(
            self.batch,
            self.n_expand,
            self.c_x,
            phi,
            x,
            b,
            self.alpha_pre,
            self.alpha_post,
            self.alpha_res,
            self.sinkhorn_repeat,
            self.sinkhorn_eps,
        )


class MHCPostWorkload(WorkloadBase):
    def __init__(self, batch: int, n_expand: int, c_x: int, dtype: torch.dtype):
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_layer_out = torch.randn([batch, c_x], device="cuda", dtype=self.dtype)
        h_post = torch.randn([batch, n_expand], device="cuda", dtype=torch.float32)
        x_res = torch.randn([batch, n_expand * c_x], device="cuda", dtype=self.dtype)
        return x_layer_out, h_post, x_res

    def ref_program(
        self, x_layer_out: torch.Tensor, h_post: torch.Tensor, x_res: torch.Tensor
    ) -> torch.Tensor:
        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            self.batch, self.n_expand * self.c_x
        ) + x_res.float()
        return x_out_ref.bfloat16()


def mhc_pre_ref(
    batch: int,
    n_expand: int,
    c_x: int,
    phi: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    alpha_pre,
    alpha_post,
    alpha_res,
    sinkhorn_repeat: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for the MHC pre-projection stage.

    Returns:
        ``(x_res_ref, x_layer_ref, h_post_ref)``: the first two bfloat16, ``h_post_ref``
        float32.
    """

    xsqr = x * x
    norm_eps = 0.0001
    r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x) + norm_eps
    H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand], device=x.device, dtype=torch.float)
    for i in range(batch):
        H[i, :] = x[i, :].float() @ phi

    H_pre_ref = H[:, :n_expand]
    H_post_ref = H[:, n_expand : 2 * n_expand]
    H_res_ref = H[:, 2 * n_expand :]
    H_res_ref = H_res_ref.reshape(batch, n_expand, n_expand)

    b_pre_ref = b[:n_expand]
    b_post_ref = b[n_expand : 2 * n_expand]
    b_res_ref = b[2 * n_expand :]
    b_res_ref = b_res_ref.reshape([n_expand, n_expand])

    H_pre_ref = torch.sigmoid(alpha_pre * H_pre_ref / r_ref.unsqueeze(-1) + b_pre_ref)
    H_post_ref = 2 * torch.sigmoid(alpha_post * H_post_ref / r_ref.unsqueeze(-1) + b_post_ref)
    H_res_ref = alpha_res * H_res_ref / r_ref.unsqueeze(-1).unsqueeze(-1) + b_res_ref

    H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

    H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
    for _i in range(sinkhorn_repeat):
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
        H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
    x_in_reshaped = x.reshape([batch, n_expand, c_x])
    x_res_ref = torch.zeros([batch, n_expand, c_x], device=x.device, dtype=torch.bfloat16)
    x_layer_ref = torch.zeros([batch, c_x], device=x.device, dtype=torch.bfloat16)

    h_res_ref = H_res_ref
    h_pre_ref = H_pre_ref
    for i in range(batch):
        h_res_tmp = h_res_ref[i, :, :].float()
        h_pre_tmp = h_pre_ref[i, :].float()
        x_in_reshaped_tmp = x_in_reshaped[i, :, :].float()
        x_res_ref[i, :, :] = h_res_tmp @ x_in_reshaped_tmp
        x_layer_ref[i, :] = h_pre_tmp @ x_in_reshaped_tmp

    x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

    x_res_ref = x_res_ref.bfloat16()
    x_layer_ref = x_layer_ref.bfloat16()
    return x_res_ref, x_layer_ref, H_post_ref
