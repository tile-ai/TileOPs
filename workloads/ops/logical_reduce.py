import torch

from workloads.base import WorkloadBase


class AnyWorkload(WorkloadBase):
    """Workload definition for AnyFwdOp.

    Generates inputs with a mix of zeros and non-zeros for meaningful
    logical reduction testing.  Boolean, integer, float, and complex
    dtypes are supported.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


class AllWorkload(WorkloadBase):
    """Workload definition for AllFwdOp.

    Same input-generation strategy as AnyWorkload (rows with all-True
    and all-False are forced for meaningful coverage).
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


class CountNonzeroWorkload(WorkloadBase):
    """Workload definition for CountNonzeroFwdOp.

    Uses the same mixed-zero input strategy as the other logical reduce
    workloads.
    """

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (_make_logical_input(self.shape, self.dtype),)


# ---------------------------------------------------------------------------
# Shared input-generation helper
# ---------------------------------------------------------------------------


def _make_logical_input(shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor with a mix of zeros and non-zeros.

    For 2-D+ inputs the first row is forced to all-zero (meaningful for
    ``any``) and the second row to all-nonzero (meaningful for ``all``).
    """
    m = shape[0] if len(shape) >= 1 else 1

    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=torch.bool, device="cuda")
        if m > 4:
            x[0] = False
            x[1] = True
    elif dtype in (torch.complex64, torch.complex128):
        real = torch.randn(*shape, dtype=torch.float32, device="cuda")
        imag = torch.randn(*shape, dtype=torch.float32, device="cuda")
        x = torch.complex(real, imag).to(dtype)
        if m > 4:
            x[0] = 0 + 0j
            x[1] = 1 + 1j
    elif dtype in (torch.int32, torch.int64):
        x = torch.randint(-5, 6, shape, dtype=dtype, device="cuda")
        if m > 4:
            x[0] = 0
            x[1] = 1
    else:
        x = torch.randn(*shape, dtype=dtype, device="cuda")
        if m > 4:
            x[0] = 0.0
            x[1] = 1.0

    return x
