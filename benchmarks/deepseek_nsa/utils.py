import torch
import tilelang


def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


def prepare_position_ids(offsets: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([torch.arange(n) for n in prepare_lens(offsets).tolist()]).to(offsets.device)


def prepare_sequence_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    return position_ids.eq(0).cumsum(0) - 1


def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(offsets)
    return torch.stack([prepare_sequence_ids(position_ids), position_ids], 1).to(offsets)


def prepare_chunk_offsets(offsets: torch.Tensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([offsets.new_tensor([0]),
                      tilelang.cdiv(prepare_lens(offsets), chunk_size)]).cumsum(-1)


def prepare_chunk_indices(offsets: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat(
        [torch.arange(n) for n in tilelang.cdiv(prepare_lens(offsets), chunk_size).tolist()])
    return torch.stack([prepare_sequence_ids(indices), indices], 1).to(offsets)
