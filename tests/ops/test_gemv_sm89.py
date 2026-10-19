"""SM89 GEMV tuning candidates fit the device shared-memory budget."""

import pytest
import torch

from tileops.kernels.gemm.dense import GemvKernel
from tileops.ops import GemmFwdOp
from workloads.gemm import GemmWorkload
from workloads.numerics import compare_outputs


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("reduce_threads,block_n", [(64, 16), (128, 8), (256, 4)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemv_largest_tuning_ring(reduce_threads: int, block_n: int, dtype: torch.dtype) -> None:
    """The largest admitted staged ring compiles and preserves GEMV outputs."""
    test = GemmWorkload(1, 256, 16384, dtype, trans_b=True)
    inputs = test.gen_inputs()
    op = GemmFwdOp(trans_b=True)
    op(*inputs)
    entries = tuple(op.built_entries("gemm").values())
    assert len(entries) == 1 and isinstance(entries[0], GemvKernel)
    kernel = entries[0]
    budget = torch.cuda.get_device_properties(kernel.device_index).shared_memory_per_block_optin
    vector_bytes = 16
    assert all(
        cfg["block_n"] * cfg["reduce_threads"] * vector_bytes * cfg["num_stages"] <= budget
        for cfg in kernel.autotune_configs
    )
    config = {"block_n": block_n, "reduce_threads": reduce_threads, "num_stages": 6}
    assert config in kernel.autotune_configs
    kernel.init_config(config)
    compare_outputs(op(*inputs), test.ref_program(*inputs), test.verification(*inputs))


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.parametrize("lhs_row", [True, False])
@pytest.mark.parametrize("operand", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gemv_offset_storage(lhs_row: bool, operand: int, dtype: torch.dtype) -> None:
    """Contiguous inputs with an element offset retain the public GEMV contract."""
    m, n = (1, 3001) if lhs_row else (3001, 1)
    test = GemmWorkload(m, n, 1024, dtype, trans_b=lhs_row)
    inputs = test.gen_inputs()
    source = inputs[operand]
    backing = torch.empty(source.numel() + 2, dtype=dtype, device=source.device)
    backing.fill_(17)
    shifted = backing[1:-1].view_as(source)
    shifted.copy_(source)
    assert shifted.is_contiguous() and shifted.storage_offset() == 1
    actual_inputs = tuple(shifted if i == operand else value for i, value in enumerate(inputs))
    before = backing.clone()
    op = GemmFwdOp(trans_b=lhs_row)
    output = op(*actual_inputs)
    entries = tuple(op.built_entries("gemm").values())
    assert entries and all(isinstance(entry, GemvKernel) for entry in entries)
    compare_outputs(output, test.ref_program(*actual_inputs), test.verification(*actual_inputs))
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
