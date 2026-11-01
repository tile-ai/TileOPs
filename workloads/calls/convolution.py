"""Call workloads of the convolution family."""

from workloads.convolution import Conv1dWorkload, Conv2dWorkload, Conv3dWorkload

WORKLOADS = {
    "Conv1dFwdOp": Conv1dWorkload.from_call,
    "Conv2dFwdOp": Conv2dWorkload.from_call,
    "Conv3dFwdOp": Conv3dWorkload.from_call,
}
