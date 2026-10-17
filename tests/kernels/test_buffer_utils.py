"""``tensors_overlap``, which the grouped GEMMs' refusal of an ``out`` overlapping an input
reads: two views of one buffer overlap exactly when they share a byte."""

import pytest
import torch

from tileops.kernels.buffer_utils import tensors_overlap

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    ("first", "second", "overlaps"),
    [
        pytest.param((0, 8), (8, 16), False, id="first-ends-where-second-starts"),
        pytest.param((8, 16), (0, 8), False, id="first-starts-where-second-ends"),
        pytest.param((0, 9), (8, 16), True, id="first-ends-one-element-in"),
        pytest.param((7, 16), (0, 8), True, id="first-starts-one-element-before-the-end"),
        pytest.param((0, 16), (4, 8), True, id="second-inside-first"),
    ],
)
def test_views_of_one_buffer_overlap_when_they_share_a_byte(first, second, overlaps):
    buffer = torch.zeros(16, dtype=torch.bfloat16)
    assert tensors_overlap(buffer[slice(*first)], buffer[slice(*second)]) is overlaps


def test_separate_buffers_never_overlap():
    assert not tensors_overlap(torch.zeros(8), torch.zeros(8))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda_only)])
@pytest.mark.parametrize("offset, overlaps", [(7, True), (8, False), (9, False)])
def test_dlpack_views_use_addresses_instead_of_storage_bases(device, offset, overlaps):
    buffer = torch.zeros(32, dtype=torch.bfloat16, device=device)
    first = torch.from_dlpack(buffer[:8])
    second = torch.from_dlpack(buffer[offset : offset + 8])
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert tensors_overlap(first, second) is overlaps
    assert tensors_overlap(second, first) is overlaps


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda_only)])
def test_empty_views_do_not_overlap(device):
    """A zero-element view overlaps no bytes in either argument order."""
    buffer = torch.zeros(32, dtype=torch.uint8, device=device)
    assert not tensors_overlap(buffer, buffer[8:8])
    assert not tensors_overlap(buffer[8:8], buffer)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda_only)])
def test_different_dtypes_use_byte_boundaries(device):
    """Overlap depends on shared bytes across element sizes, including one byte."""
    buffer = torch.zeros(32, dtype=torch.uint8, device=device)
    floats = buffer.view(torch.float32)
    assert tensors_overlap(floats[:2], buffer[7:9])
    assert not tensors_overlap(floats[:2], buffer[8:])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda_only)])
def test_dense_transposes_include_the_last_byte(device):
    """A dense transpose overlaps a view of its last byte."""
    buffer = torch.zeros(32, dtype=torch.uint8, device=device)
    floats = buffer.view(torch.float32)
    assert tensors_overlap(floats.view(2, 4).T, buffer[31:])


def test_dense_dlpack_overlap_matches_enumerated_bytes():
    """All slices of a small allocation agree with an independent byte-set oracle."""
    buffer = torch.zeros(4, dtype=torch.uint8)
    views = [
        (torch.from_dlpack(buffer[start:end]), set(range(start, end)))
        for start in range(buffer.numel() + 1)
        for end in range(start, buffer.numel() + 1)
    ]
    for first, first_bytes in views:
        for second, second_bytes in views:
            assert tensors_overlap(first, second) is bool(first_bytes & second_bytes)


@pytest.mark.cuda_only
def test_tensors_on_different_devices_do_not_overlap():
    """CPU and CUDA tensors occupy different address spaces."""
    cpu = torch.zeros(8)
    cuda = cpu.to("cuda")
    assert not tensors_overlap(cpu, cuda)
    assert not tensors_overlap(cuda, cpu)
