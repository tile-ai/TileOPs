import torch


def quantize_kv_fa3_descale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x / descale[:, None, :, None], -448.0,
                        448.0).to(torch.float8_e4m3fn).contiguous()
    return x_fp8, descale.float().contiguous()


def quantize_q_fa3_gqa_descale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    return x_fp8.reshape(batch, seq_len, heads, dim).contiguous(), descale.float().contiguous()
