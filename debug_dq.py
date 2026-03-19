"""Debug script: compare intra kernel dq intermediate values with torch ref."""
import torch

from tileops.kernels.gla_chunkwise.gla_bwd import _gla_bwd_intra_kernel, _gla_precompute_g_kernel

torch.manual_seed(42)
B, T, H, K, V, BC = 1, 64, 1, 64, 64, 64
dtype = torch.bfloat16
scale = K ** -0.5

q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1


def cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


# --- Torch reference ---
gc = g.float().cumsum(dim=1)[0, :, 0, :]  # [BC, K]
qc = q[0, :, 0, :].float()
kc = k[0, :, 0, :].float()
vc = v[0, :, 0, :].float()
doc = do[0, :, 0, :].float()

# A (attention with gating)
A_ref = torch.zeros(BC, BC, device="cuda")
for ik in range(K):
    for i in range(BC):
        for j in range(i + 1):
            A_ref[i, j] += qc[i, ik] * kc[j, ik] * torch.exp(gc[i, ik] - gc[j, ik])
A_ref *= scale

# dA
da_ref = torch.tril(scale * doc @ vc.T)

# dv_intra = A^T @ do
dv_ref = A_ref.T @ doc

# dq_intra
dq_ref = torch.zeros(BC, K, device="cuda")
for j in range(BC):
    for i in range(j, BC):
        dq_ref[i] += da_ref[i, j] * kc[j] * torch.exp(gc[i] - gc[j])

# dk_intra
dk_ref = torch.zeros(BC, K, device="cuda")
for i in range(BC):
    for j in range(i + 1):
        dk_ref[j] += da_ref[i, j] * qc[i] * torch.exp(gc[i] - gc[j])

# --- TileOPs kernel ---
g_fn = _gla_precompute_g_kernel(B, T, H, K, BC, "bfloat16")(2, 128)
g_cumsum_op = g_fn(g)

print(f"g_cumsum: cos={cos(gc, g_cumsum_op[0,:,0,:]):.6f}")

for dtype_name, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
    intra_fn = _gla_bwd_intra_kernel(B, T, H, K, V, BC, scale, dtype_name)(2, 128)
    # Need to pass inputs in the kernel's dtype
    if dt == torch.float32:
        dq_op, dk_op, dv_op = intra_fn(q.float(), k.float(), v.float(), g_cumsum_op, do.float())
    else:
        dq_op, dk_op, dv_op = intra_fn(q, k, v, g_cumsum_op, do)

    dq_v = dq_op[0, :, 0, :].float()
    dk_v = dk_op[0, :, 0, :].float()
    dv_v = dv_op[0, :, 0, :].float()

    print(f"\n--- {dtype_name} ---")
    print(f"dv: cos={cos(dv_ref, dv_v):.6f}")
    print(f"dq: cos={cos(dq_ref, dq_v):.6f}")
    print(f"dk: cos={cos(dk_ref, dk_v):.6f}")

    if cos(dq_ref, dq_v) < 0.99:
        # Show per-row detail
        for row in [0, 1, 4, 8, 16, 32, 48, 63]:
            c = cos(dq_ref[row], dq_v[row])
            print(f"  dq row {row:2d}: cos={c:.4f}  ref_l2={dq_ref[row].norm():.4e}  op_l2={dq_v[row].norm():.4e}")
