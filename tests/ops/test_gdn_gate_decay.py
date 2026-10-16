import pytest
import torch

from tests.workload_test_base import TestBase
from tileops.ops import GDNFwdOp
from workloads.linear_attention.gdn import GDNFwdWorkload

pytestmark = pytest.mark.smoke

LOG_DECAY = -4.0
SEED = 42


class GDNGateDecayTest(GDNFwdWorkload, TestBase):
    pass


@pytest.mark.parametrize(
    "_architecture",
    [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)],
)
@pytest.mark.parametrize("length", [16, 65], ids=["partial-subblock", "cross-chunk"])
def test_gdn_prefill_preserves_finite_strong_decay(_architecture: int, length: int) -> None:
    """Strong decay must not divide underflowed gates or overflow centered FP16 scaling."""
    torch.manual_seed(SEED)
    test = GDNGateDecayTest(
        batch=1,
        seq_len=length,
        heads=2,
        dim=128,
        dtype=torch.float16,
        has_initial_state=True,
    )
    inputs = test.gen_inputs()
    gate = torch.full_like(inputs[3], LOG_DECAY)
    test.check(GDNFwdOp(), *inputs[:3], gate, *inputs[4:])
