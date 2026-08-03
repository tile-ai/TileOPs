"""Correctness test for the top-k selector op.

Compares kernel output against torch.topk using set-intersection
(topk indices may appear in different order for equal-valued elements).
"""

from __future__ import annotations

import pytest
import torch

from manifest import load_workloads
from ops import TopkSelectorOp
from benchmarks.benchmark_base import workload_field_params
from workloads.topk_selector import TopkSelectorWorkload

_PARAMS = workload_field_params(
    load_workloads("TopkSelectorOp"),
    ("batch", "seq_len", "seq_len_kv", "kv_group", "topk", "in_dtype", "out_dtype"),
)


def _set_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    ref_np = output_ref.cpu().to(torch.int32).numpy()
    trt_np = output.cpu().to(torch.int32).numpy()
    set_ref = set(ref_np.flatten().tolist())
    set_trt = set(trt_np.flatten().tolist())
    intersection = set_ref & set_trt
    assert len(intersection) / len(set_ref) == 1.0, \
        "output indices do not match reference indices"


@pytest.mark.parametrize(
    "batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype",
    _PARAMS,
)
def test_topk_selector_op(batch: int, seq_len: int, seq_len_kv: int,
                          kv_group: int, topk: int,
                          in_dtype: torch.dtype, out_dtype: torch.dtype) -> None:
    wl = TopkSelectorWorkload(batch, seq_len, seq_len_kv, kv_group,
                              topk, in_dtype, out_dtype)
    inputs = wl.gen_inputs()
    op = TopkSelectorOp(topk=topk, tune=False)
    output = op(*inputs)

    index_score = inputs[0]
    ref = torch.topk(index_score, topk, dim=2)[1].permute(0, 1, 3, 2)
    _set_compare(output, ref)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
