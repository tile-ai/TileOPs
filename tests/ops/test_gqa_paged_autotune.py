"""Packed paged attention must supply the complete tuning call."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.kernels.attention.gqa.paged import GQAPagedFwdKernel
from tileops.ops import GQAPagedFwdOp
from workloads.attention.gqa.paged import GQAPagedFwdWorkload


@pytest.mark.smoke
@pytest.mark.sm89
@pytest.mark.in_tree_kernels
def test_gqa_paged_autotune_supplies_packed_inputs(monkeypatch) -> None:
    """Guard the supplier's three missing scale inputs through real Op autotuning.

    SM89 selects the ordinary paged kernel here; SM90 can select another implementation.
    """
    production_configs = GQAPagedFwdKernel.autotune_configs.fget

    def one_config(entry):
        candidates = production_configs(entry)
        return [
            next(
                config
                for config in candidates
                if config["block_M"] == 64
                and config["block_N"] == 64
                and config["num_stages"] == 1
                and config["threads"] == 128
            )
        ]

    # Keep the initial seed in the same single-stage compilation path as the sweep.
    monkeypatch.setattr(GQAPagedFwdKernel, "default_config", property(lambda e: one_config(e)[0]))
    monkeypatch.setattr(GQAPagedFwdKernel, "autotune_configs", property(one_config))
    workload = GQAPagedFwdWorkload(32, 8, 128, [160, 160], [700, 450], 64, 11, 22, torch.float16)
    op = GQAPagedFwdOp()
    op.request_tune()
    TestBase.check(workload, op, *workload.gen_inputs())
    entries = [
        entry for interface in op.interfaces for entry in op.built_entries(interface).values()
    ]
    assert entries and all(isinstance(entry, GQAPagedFwdKernel) for entry in entries)
