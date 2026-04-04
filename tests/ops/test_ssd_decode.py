
from tests.test_base import TestBase, allclose_compare
from tileops.ops.ssd_decode import SsdDecodeOp
from workloads.ops.ssd_decode import SsdDecodeFixture, ssd_decode_ref
from workloads.ops.ssd_decode import SsdDecodeTest as _SsdDecodeTestWorkload


class SsdDecodeTest(_SsdDecodeTestWorkload, TestBase):
    def ref_program(self, A, dt, x, B_in, C_in, state):
        return ssd_decode_ref(A, dt, x, B_in, C_in, state)


@SsdDecodeFixture
def test_ssd_decode(batch, n_heads, d_head, d_state, n_groups, dtype, tune):
    test = SsdDecodeTest(batch, n_heads, d_head, d_state, n_groups, dtype)
    op = SsdDecodeOp(batch, n_heads, d_head, d_state, n_groups, dtype, tune=tune)
    A, dt, x, B_in, C_in, state = test.gen_inputs()

    # Run reference on a clone of state so the two runs start from the same point.
    state_ref = state.clone()
    y_ref = test.ref_program(A, dt, x, B_in, C_in, state_ref)

    # Run kernel; state is updated in-place.
    y_op = op(A, dt, x, B_in, C_in, state)

    atol = 1e-3
    rtol = 1e-3
    allclose_compare(y_op, y_ref, atol=atol, rtol=rtol)
    allclose_compare(state, state_ref, atol=atol, rtol=rtol)
