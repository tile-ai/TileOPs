from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba import SSDStatePassingFwdKernel

from ..op_base import Op

__all__ = ["SSDStatePassingFwdOp"]


class SSDStatePassingFwdOp(Op):
    """Mamba-2 State-Space Dual (SSD) state passing forward operator.

    Performs the inter-chunk recurrent scan:

      s_c[m] = exp(dA_chunk_cumsum[b, h, c]) * s_{c-1}[m] + states[b, c, h, m]

    with s_{-1} = initial_states, or 0 when it is not passed.

    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "ssd_state_passing_fwd": SSDStatePassingFwdKernel
    }

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune the tile config when a kernel is first built.
        """
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, initial-state presence, dtype and device."""
        batch, num_chunks, n_heads, d_state, has_initial_states, dtype, _device = call
        return call, lambda: self.kernel_map["ssd_state_passing_fwd"](
            batch,
            num_chunks,
            n_heads,
            d_state,
            has_initial_states=has_initial_states,
            dtype=dtype,
            tune=self.tune,
        )

    def forward(
        self,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the SSD state passing forward pass.

        Args:
            states:           (batch, num_chunks, n_heads, d_state)
            dA_chunk_cumsum:  (batch, n_heads, num_chunks) float32
            initial_states:   (batch, n_heads, d_state) float32, optional

        Returns:
            prev_states:  (batch, num_chunks, n_heads, d_state) float32
            final_states: (batch, n_heads, d_state) float32
        """
        return self._call_boundary(states, dA_chunk_cumsum, initial_states)

    def _eager_forward(
        self,
        states: torch.Tensor,
        dA_chunk_cumsum: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, num_chunks, n_heads, d_state = states.shape
        kernel = self.kernel_for(
            "ssd_state_passing_fwd",
            (states, dA_chunk_cumsum, initial_states),
            (
                batch,
                num_chunks,
                n_heads,
                d_state,
                initial_states is not None,
                states.dtype,
                states.device.index,
            ),
        )

        states = states.contiguous()
        dA_chunk_cumsum = dA_chunk_cumsum.contiguous()
        if initial_states is None:
            # The kernel built for this call starts from zero, so this buffer
            # only fills the argument slot and is never read.
            initial_states = states.new_empty(batch, n_heads, d_state, dtype=torch.float32)
        else:
            initial_states = initial_states.contiguous()

        return kernel(states, dA_chunk_cumsum, initial_states)
