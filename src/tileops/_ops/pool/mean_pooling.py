from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.pool import (
    MeanPoolingCall,
    MeanPoolingFwdInterface,
    MeanPoolingFwdKernel,
)
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["MeanPoolingFwdOp"]


class MeanPoolingFwdOp(Op):
    """Chunked mean over the sequence axis of a ``[batch, seq, heads, dim]`` tensor.

    Not a PyTorch pooling op, and torch has no counterpart. The sequence axis is cut into
    chunks of ``chunk_size`` and each chunk is averaged, giving one output row per chunk.
    Pass ``cu_seqlens`` and ``chunk_indices`` and the chunks follow the ragged sequence
    boundaries ``cu_seqlens`` describes instead of a uniform split; ``chunk_indices`` then
    names the ``(sequence, chunk-within-sequence)`` pair each output row belongs to, one row
    per chunk.

    A sequence's last chunk may be shorter than ``chunk_size``. It is divided by the count
    it actually holds, so no padding is averaged in. On a uniform split that makes the op
    equal to ``torch.nn.functional.avg_pool1d(kernel_size=chunk_size, stride=chunk_size,
    ceil_mode=True)`` over a view with the sequence axis last. Chunk sums accumulate in
    ``accum_dtype`` and are cast back to the input dtype at the boundary, so a ``float16``
    input with ``accum_dtype=torch.float32`` does not lose the sum to rounding.

    By default the op does not check the contents of ``cu_seqlens`` and ``chunk_indices``.
    The kernel indexes ``cu_seqlens`` by ``chunk_indices`` and ``x`` by ``cu_seqlens``
    without bounds checks, so an inconsistent map can cause out-of-bounds accesses or
    incorrect results. The caller guarantees a consistent map, or passes
    ``validate_inputs=True``, which checks it on every call at the cost of device
    synchronizations and cannot run inside CUDA Graph capture.

    Example:
        ```python linenums="1"
        op = MeanPoolingFwdOp(chunk_size=64, accum_dtype=torch.float32)
        chunk_means = op(x)  # uniform split
        chunk_means = op(x, cu_seqlens, chunk_indices)  # ragged, one index row per chunk
        ```
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "mean_pooling_fwd_kernel": MeanPoolingFwdKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "mean_pooling": MeanPoolingFwdInterface
    }

    def __init__(
        self,
        chunk_size: int,
        accum_dtype: torch.dtype,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            chunk_size: Manifest ``params.chunk_size``, ``int``, a positive multiple of 32.
            accum_dtype: Manifest ``params.accum_dtype``, ``torch.dtype`` — what a chunk sum
                accumulates in.
            validate_inputs: Check ragged metadata contents synchronously on each call.
                Disable during CUDA Graph capture; shapes and dtypes are always checked.
            target: Backend target to serve this op, or ``None`` to decide from the input
                device.
        """
        self.validate_inputs = validate_inputs
        self.chunk_size = chunk_size
        self.accum_dtype = accum_dtype
        # Keyed by (device, shape): the uniform path hands the kernel tensors it never
        # reads, and a placeholder on the wrong device would route the launch there.
        self._placeholders: Dict[tuple, torch.Tensor] = {}
        super().__init__(target=target)

    def _placeholder(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        key = (device, shape)
        if key not in self._placeholders:
            self._placeholders[key] = torch.zeros(shape, dtype=torch.int32, device=device)
        return self._placeholders[key]

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Average each chunk of ``x``'s sequence axis.

        Args:
            x: Input tensor, ``[batch, seq, heads, dim]``, dtype ``float16``, ``bfloat16``
                or ``float32``. ``dim`` is at most 128 or a multiple of it, which is what the
                manifest declares rather than what the kernel needs.
            cu_seqlens: Sequence boundaries, ``[seq_num + 1]``, dtype ``int32``. Passing it
                selects the ragged split, and it comes with ``chunk_indices``.
            chunk_indices: One ``(sequence, chunk-within-sequence)`` pair per output chunk,
                ``[chunks, 2]``, dtype ``int32``.

        Returns:
            ``output``, ``[batch, chunks, heads, dim]``, dtype as ``x``. ``chunks`` is
            ``ceil(seq / chunk_size)`` for a uniform split and ``chunk_indices.shape[0]`` for a
            ragged one.

        Raises:
            ValueError: With ``validate_inputs=True``, ``chunk_indices`` disagrees with
                ``cu_seqlens`` or ``cu_seqlens`` does not partition the sequence axis.
        """
        x = x.contiguous()
        batch_size, seq_len, heads, dim = x.shape
        ragged = cu_seqlens is not None
        if ragged:
            chunks = chunk_indices.shape[0]
            if self.validate_inputs:
                self._check_ragged(cu_seqlens, chunk_indices, seq_len, chunks)
            seq_num = cu_seqlens.shape[0] - 1
            cu_seqlens_arg, chunk_indices_arg = cu_seqlens.contiguous(), chunk_indices.contiguous()
        else:
            chunks = -(-seq_len // self.chunk_size)
            # The kernel takes both tensors whether or not it reads them; `inputs` keeps
            # the caller's `None`s, which is where presence is read from. `seq_num = 0`
            # divides by zero in the autotune supply, so one whole-axis sequence it is.
            seq_num = 1
            cu_seqlens_arg = self._placeholder((2,), x.device)
            chunk_indices_arg = self._placeholder((chunks, 2), x.device)

        kernel = self.kernel_for(
            "mean_pooling",
            lambda: MeanPoolingCall(
                batch_size=batch_size,
                seq_len=seq_len,
                heads=heads,
                dim=dim,
                chunk_size=self.chunk_size,
                chunks_per_batch=chunks,
                seq_num=seq_num,
                use_offsets=ragged,
                dtype=x.dtype,
                accum_dtype=self.accum_dtype,
                device=x.device,
            ),
        )
        return kernel(x, cu_seqlens_arg, chunk_indices=chunk_indices_arg)

    def _check_ragged(
        self, cu_seqlens: torch.Tensor, chunk_indices: torch.Tensor, seq_len: int, chunks: int
    ) -> None:
        """Check `chunk_indices` against `cu_seqlens` rather than believing its row count.

        The output's chunk axis has to come from a shape, because that is all the compile
        fake is handed, so it comes from `chunk_indices`. The values are here, so this is where
        a `chunk_indices` that disagrees with `cu_seqlens` is caught.
        """
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if int(lengths.min()) < 0:
            raise ValueError("cu_seqlens must be non-decreasing")
        # A partition, not a window: every token belongs to exactly one sequence.
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != seq_len:
            raise ValueError(
                f"cu_seqlens must run 0 to x's sequence axis of {seq_len}; got "
                f"{int(cu_seqlens[0])} to {int(cu_seqlens[-1])}"
            )
        per_seq = -(-lengths // self.chunk_size)
        implied = int(per_seq.sum())
        if chunks != implied:
            raise ValueError(
                f"chunk_indices holds {chunks} chunks but cu_seqlens imply {implied} for "
                f"chunk_size={self.chunk_size}"
            )
        # The kernel reads the chunk each row names, in range or not.
        seq_ids, chunk_ids = chunk_indices[:, 0].long(), chunk_indices[:, 1]
        if int(seq_ids.min()) < 0 or int(seq_ids.max()) >= per_seq.shape[0]:
            raise ValueError(
                f"chunk_indices names a sequence outside cu_seqlens' {per_seq.shape[0]} sequences"
            )
        if not bool(((chunk_ids >= 0) & (chunk_ids < per_seq[seq_ids])).all()):
            raise ValueError("chunk_indices names a chunk its sequence does not have")
        # A repeat would emit one chunk twice and drop another.
        chunk_base = per_seq.cumsum(0) - per_seq
        if int(torch.unique(chunk_base[seq_ids] + chunk_ids).numel()) != chunks:
            raise ValueError("chunk_indices must name each chunk cu_seqlens implies exactly once")
