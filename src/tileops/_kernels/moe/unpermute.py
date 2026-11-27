"""Weighted inverse permute for staged MoE PostPermute.

Inputs:
  expert_output    [materialized_rows, H] bf16/fp16 expert output
  inverse_indices  [T*K]                  int32 flat route → materialized row
  topk_weights     [T, K]                 float32 routing weights
  residual         [T, H]                 bf16/fp16, optional, added before the cast

Output:
  output           [T, H]                 bf16/fp16
"""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import CUDA_COMPILE_FLAGS, target_cache
from tileops._kernels.constants import MAX_BLOCK_THREADS, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.moe.call_spec import PostPermuteCall, PostPermuteFwdInterface
from tileops._kernels.moe.layout import permute_refusal, row_threads

__all__ = ["MoEUnpermuteKernel"]


@target_cache(maxsize=32)
def _make_unpermute_kernel(
    num_tokens: int,
    top_k: int,
    hidden_size: int,
    materialized_rows: int,
    dtype: str,
    threads: int,
    scaling: float,
    has_residual: bool,
):
    """Build one block per output token with an fp32 reduction.

    With *has_residual* the program takes a ``(num_tokens, hidden_size)`` residual after the
    routing weights and adds it to the scaled reduction before the cast.
    """
    numel = num_tokens * top_k

    @tilelang.jit(out_idx=[], compile_flags=[*CUDA_COMPILE_FLAGS, "-DENABLE_BF16"])
    def _unpermute():
        @T.macro
        def reduce_tokens(expert_output, inverse_indices, topk_weights, residual, output):
            with T.Kernel(num_tokens, threads=threads) as (token_idx,):
                acc = T.alloc_fragment([hidden_size], "float32")
                src = T.alloc_fragment([hidden_size], dtype)
                if residual is not None:
                    res = T.alloc_fragment([hidden_size], dtype)
                    T.copy(residual[token_idx, 0:hidden_size], res)

                T.fill(acc, 0.0)

                # num_stages=2 overlaps the latency-bound gathers; top_k < 2 leaves
                # the pipeline deeper than the trip count, so it runs serial.
                for k in T.Pipelined(top_k, num_stages=2) if top_k >= 2 else T.serial(top_k):
                    flat_idx = token_idx * T.int32(top_k) + k
                    slot = inverse_indices[flat_idx]
                    weight = topk_weights[token_idx, k]
                    T.copy(expert_output[slot, 0:hidden_size], src)
                    for j in T.Parallel(hidden_size):
                        acc[j] = acc[j] + T.Cast("float32", src[j]) * weight

                out_frag = T.alloc_fragment([hidden_size], dtype)
                if residual is not None:
                    for j in T.Parallel(hidden_size):
                        out_frag[j] = T.Cast(
                            dtype, acc[j] * T.float32(scaling) + T.Cast("float32", res[j])
                        )
                elif scaling != 1.0:
                    for j in T.Parallel(hidden_size):
                        out_frag[j] = T.Cast(dtype, acc[j] * T.float32(scaling))
                else:
                    for j in T.Parallel(hidden_size):
                        out_frag[j] = T.Cast(dtype, acc[j])
                T.copy(out_frag, output[token_idx, 0:hidden_size])

        if has_residual:

            @T.prim_func
            def _unpermute_residual_main(
                expert_output: T.Tensor([materialized_rows, hidden_size], dtype),
                inverse_indices: T.Tensor([numel], "int32"),
                topk_weights: T.Tensor([num_tokens, top_k], "float32"),
                residual: T.Tensor([num_tokens, hidden_size], dtype),
                output: T.Tensor([num_tokens, hidden_size], dtype),
            ):
                reduce_tokens(expert_output, inverse_indices, topk_weights, residual, output)

            return _unpermute_residual_main

        @T.prim_func
        def _unpermute_main(
            expert_output: T.Tensor([materialized_rows, hidden_size], dtype),
            inverse_indices: T.Tensor([numel], "int32"),
            topk_weights: T.Tensor([num_tokens, top_k], "float32"),
            output: T.Tensor([num_tokens, hidden_size], dtype),
        ):
            reduce_tokens(expert_output, inverse_indices, topk_weights, None, output)

        return _unpermute_main

    return _unpermute


class MoEUnpermuteKernel(Kernel, PostPermuteFwdInterface):
    """Weighted inverse-permute kernel for staged PostPermute.

    Args:
        num_tokens: Number of input tokens T.
        top_k: Number of experts selected per token K.
        hidden_size: Hidden dimension H.
        materialized_rows: Number of materialized expert-output rows.
        scaling: Scalar multiplied into the reduced output before the cast/store
            (folds ``routed_scaling_factor``). Defaults to 1.0 (no scaling).
        has_residual: Whether each call passes a residual added to the scaled reduction
            before the cast.
        dtype: Data type of expert output and final output (bf16 or fp16).
        sm_count: Device SM count used to choose the low-token launch width.

    Example:
        ```python linenums="1"
        kernel = MoEUnpermuteKernel(num_tokens=4, top_k=2, hidden_size=128, materialized_rows=8)
        output = kernel(expert_output, inverse_indices, topk_weights)
        ```
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: PostPermuteCall) -> "str | None":
        reason = permute_refusal(call.layout_key, call.input_dtype)
        if reason is not None:
            return reason
        if call.output_dtype != call.input_dtype:
            return f"writes {call.input_dtype}, not {call.output_dtype}"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: PostPermuteCall) -> Entry:
        """One build per route extents, hidden width, routing scale, residual presence, dtype
        and device."""
        index = call.device_index
        args = (
            call.num_tokens,
            call.top_k,
            call.hidden_size,
            call.materialized_rows,
            call.epilogue.routed_scaling_factor,
            call.has_residual,
            call.input_dtype,
        )
        return (*args, index), lambda: cls(*args, sm_count=call.sm_count, device_index=index)

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        materialized_rows: int,
        scaling: float = 1.0,
        has_residual: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        *,
        sm_count: int | None = None,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.materialized_rows = materialized_rows
        self.dtype = dtype
        self.sm_count = self.build_target.sm_count if sm_count is None else sm_count
        if self.sm_count <= 0:
            raise ValueError("sm_count must be positive")
        self.scaling = scaling
        self.has_residual = has_residual
        self.init_config()

    @property
    def default_config(self) -> dict:
        # The largest width that divides the row, whose fragment is laid out over the
        # threads, at one vector access a thread: up to a block's thread limit with no more
        # tokens than SMs, up to 256 otherwise so several blocks share an SM.
        cap = MAX_BLOCK_THREADS if self.num_tokens <= self.sm_count else 256
        vector = VECTOR_ACCESS_BYTES // self.dtype.itemsize
        return {"threads": row_threads(self.hidden_size, vector, cap)}

    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run moe_unpermute.

        Args:
            expert_output: [materialized_rows, H] bf16/fp16 expert output.
            inverse_indices: [T*K] int32 flat route → materialized row.
            topk_weights: [T, K] float32 routing weights.
            out: optional [T, H] output buffer to write into and reuse across
                calls. Must not overlap any input. Allocated internally with
                ``torch.empty`` if omitted.
            residual: [T, H] bf16/fp16, added to the scaled reduction before the cast;
                passed exactly when the kernel was built with ``has_residual``.

        Returns:
            output: [T, H] bf16/fp16 (``out`` if provided).
        """
        assert inverse_indices.dtype == torch.int32
        assert topk_weights.dtype == torch.float32
        self._require_devices(expert_output=expert_output, residual=residual)

        dev = expert_output.device
        if out is None:
            output = torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device=dev)
        else:
            if tuple(out.shape) != (self.num_tokens, self.hidden_size):
                raise ValueError(
                    f"out shape must be {(self.num_tokens, self.hidden_size)}, "
                    f"got {tuple(out.shape)}"
                )
            if out.dtype != self.dtype:
                raise ValueError(f"out dtype must be {self.dtype}, got {out.dtype}")
            # A cross-device or non-contiguous ``out`` would scatter the store.
            if out.device != dev:
                raise ValueError(f"out device must be {dev}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        inputs = (expert_output, inverse_indices, topk_weights)
        if self.has_residual:
            inputs += (residual,)
        self.program(self._compile)(*inputs, output)

        return output

    def _compile(self, cfg: dict):
        """The program of config *cfg*."""
        return _make_unpermute_kernel(
            self.num_tokens,
            self.top_k,
            self.hidden_size,
            self.materialized_rows,
            self.dtype_str,
            cfg["threads"],
            self.scaling,
            self.has_residual,
        )()
