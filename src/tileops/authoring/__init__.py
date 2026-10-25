"""What code outside TileOPs uses to write a new op on the `Op` base class.

A new op subclasses `Op` and states its manifest entry as the class attribute `spec`. Each
place it calls a kernel is a `KernelInterface`, implemented by `Kernel` subclasses;
`Op.kernel_for` selects one for a `CallSpec` and stores the build as an `Entry`. `Op.last_call`
is the `SignatureCall` of the op's last completed call. `tileops.Op` and `tileops.authoring.Op`
are the same class.
"""

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops._ops._signature_codegen import SignatureCall
from tileops._ops.op_base import Op

__all__ = ["CallSpec", "Entry", "Kernel", "KernelInterface", "Op", "SignatureCall"]
