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
