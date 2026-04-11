from workloads.base import RandnTest


class SoftmaxTest(RandnTest):
    """Workload definition for SoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSoftmaxTest(RandnTest):
    """Workload definition for LogSoftmaxFwdOp (spec interface: shape + dtype)."""


class LogSumExpTest(RandnTest):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""
