from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.pool import (
    MeanPoolingCall,
    MeanPoolingFwdInterface,
    MeanPoolingFwdKernel,
)
from tileops.ops.op_base import Op

__all__ = ["MeanPoolingFwdOp"]


class MeanPoolingFwdOp(Op):
    """Chunked mean over the sequence axis of a ``[batch, seq, heads, dim]`` tensor.

    Not a PyTorch pooling op, and torch has no counterpart. The sequence axis is cut into
    chunks of ``chunk_size`` and each chunk is averaged, giving one output row per chunk.
    Pass ``offsets`` and ``indices`` and the chunks follow the ragged sequence boundaries
    ``offsets`` describes instead of a uniform split; ``indices`` then names the
    ``(sequence, chunk-within-sequence)`` pair each output row belongs to, one row per
    chunk.

    A sequence's last chunk may be shorter than ``chunk_size``. It is divided by the count
    it actually holds, so no padding is averaged in. On a uniform split that makes the op
    equal to ``torch.nn.functional.avg_pool1d(kernel_size=chunk_size, stride=chunk_size,
    ceil_mode=True)`` over a view with the sequence axis last. Chunk sums accumulate in
    ``accum_dtype`` and are cast back to the input dtype at the boundary, so a ``float16``
    input with ``accum_dtype=torch.float32`` does not lose the sum to rounding.

    By default the op does not check the contents of ``offsets`` and ``indices``. The
    kernel indexes ``offsets`` by ``indices`` and ``x`` by ``offsets`` without bounds
    checks, so a map that does not partition the sequence axis reads device memory out of
    bounds and returns undefined values. The caller guarantees a consistent map, or passes
    ``validate_inputs=True``, which checks it on every call at the cost of device
    synchronizations and cannot run inside CUDA Graph capture.

    Example:
        ```python linenums="1"
        op = MeanPoolingFwdOp(chunk_size=64, accum_dtype=torch.float32)
        chunk_means = op(x)                        # uniform split
        chunk_means = op(x, offsets, indices)      # ragged, one row of indices per chunk
        ```
    """

    compile_boundary: ClassVar[bool] = True

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
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
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
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.validate_inputs = validate_inputs
        self.chunk_size = chunk_size
        self.accum_dtype = accum_dtype
        self.target = target
        self.tune = tune
        # Keyed by (device, shape): the uniform path hands the kernel tensors it never
        # reads, and a placeholder on the wrong device would route the launch there.
        self._placeholders: Dict[tuple, torch.Tensor] = {}
        self.dispatch_kernel(kernel_map)

    def _placeholder(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        key = (device, shape)
        if key not in self._placeholders:
            self._placeholders[key] = torch.zeros(shape, dtype=torch.int32, device=device)
        return self._placeholders[key]

    def forward(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Average each chunk of ``x``'s sequence axis.

        Args:
            x: Input tensor, ``[batch, seq, heads, dim]``, dtype ``float16``, ``bfloat16``
                or ``float32``. ``dim`` is at most 128 or a multiple of it, which is what the
                manifest declares rather than what the kernel needs.
            offsets: Sequence boundaries, ``[seq_num + 1]``, dtype ``int32``. Passing it
                selects the ragged split, and it comes with ``indices``.
            indices: One ``(sequence, chunk-within-sequence)`` pair per output chunk,
                ``[chunks, 2]``, dtype ``int32``.

        Returns:
            ``output``, ``[batch, chunks, heads, dim]``, dtype as ``x``. ``chunks`` is
            ``ceil(seq / chunk_size)`` for a uniform split and ``indices.shape[0]`` for a
            ragged one.

        Raises:
            ValueError: With ``validate_inputs=True``, ``indices`` disagrees with
                ``offsets`` or ``offsets`` does not partition the sequence axis.
        """
        return self._call_boundary(x, offsets, indices)

    def _eager_forward(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator."""
        # Heads and dim are read as one width.
        x = x.contiguous()
        batch_size, seq_len, heads, dim = x.shape
        ragged = offsets is not None
        if ragged:
            chunks = indices.shape[0]
            if self.validate_inputs:
                self._check_ragged(offsets, indices, seq_len, chunks)
            seq_num = offsets.shape[0] - 1
            offsets_arg, indices_arg = offsets.contiguous(), indices.contiguous()
        else:
            chunks = -(-seq_len // self.chunk_size)
            # The kernel takes both tensors whether or not it reads them; `inputs` keeps
            # the caller's `None`s, which is where presence is read from. `seq_num = 0`
            # divides by zero in the autotune supply, so one whole-axis sequence it is.
            seq_num = 1
            offsets_arg = self._placeholder((2,), x.device)
            indices_arg = self._placeholder((chunks, 2), x.device)

        call = MeanPoolingCall(
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
        )
        kernel = self.kernel_for("mean_pooling", call)
        return kernel(x, offsets_arg, indices=indices_arg)

    def _check_ragged(
        self, offsets: torch.Tensor, indices: torch.Tensor, seq_len: int, chunks: int
    ) -> None:
        """Check `indices` against `offsets` rather than believing its row count.

        The output's chunk axis has to come from a shape, because that is all the compile
        fake is handed, so it comes from `indices`. The values are here, so this is where
        an `indices` that disagrees with `offsets` is caught.
        """
        lengths = offsets[1:] - offsets[:-1]
        if int(lengths.min()) < 0:
            raise ValueError("offsets must be non-decreasing")
        # A partition, not a window: every token belongs to exactly one sequence.
        if int(offsets[0]) != 0 or int(offsets[-1]) != seq_len:
            raise ValueError(
                f"offsets must run 0 to x's sequence axis of {seq_len}; got "
                f"{int(offsets[0])} to {int(offsets[-1])}"
            )
        per_seq = -(-lengths // self.chunk_size)
        implied = int(per_seq.sum())
        if chunks != implied:
            raise ValueError(
                f"indices holds {chunks} chunks but offsets imply {implied} for "
                f"chunk_size={self.chunk_size}"
            )
        # The kernel reads the chunk each row names, in range or not.
        seq_ids, chunk_ids = indices[:, 0].long(), indices[:, 1]
        if int(seq_ids.min()) < 0 or int(seq_ids.max()) >= per_seq.shape[0]:
            raise ValueError(
                f"indices names a sequence outside offsets' {per_seq.shape[0]} sequences"
            )
        if not bool(((chunk_ids >= 0) & (chunk_ids < per_seq[seq_ids])).all()):
            raise ValueError("indices names a chunk its sequence does not have")
        # A repeat would emit one chunk twice and drop another.
        chunk_base = per_seq.cumsum(0) - per_seq
        if int(torch.unique(chunk_base[seq_ids] + chunk_ids).numel()) != chunks:
            raise ValueError("indices must name each chunk offsets implies exactly once")
