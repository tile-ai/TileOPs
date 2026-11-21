"""Gated DeltaNet (GDN) inference decode: one token a sequence, or each sequence's tokens
in turn when verifying drafts on a state pool."""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.linear_attention.call_spec import (
    GDNCall,
    GDNFwdInterface,
    head_count_refusal,
)
from tileops._kernels.linear_attention.delta_decode import decode_launch, delta_decode_tl

__all__ = ["GDNColumnDecodeKernel", "GDNDenseDecodeFwdKernel"]


class GDNDenseDecodeFwdKernel(Kernel, GDNFwdInterface):
    """FP16/BF16 decode with FP32 recurrent state.

    The decayed state slice stays in registers, so the state update and output
    projection reuse the same load. The gate transform, the beta sigmoid, the Q/K
    L2 normalization and the starting state are build flags of the program, so a
    call that asks for none of them runs a program that contains none of them.
    """

    supported_archs = [80, 89, 90]

    @classmethod
    def refusal(cls, call: GDNCall) -> str | None:
        """Why this kernel does not serve *call*, or ``None`` when it does.

        One token continuing a 64- or 128-wide square state, in either layout, in batch order
        or in the slots of a state pool, under any combination of the recurrence flags the
        operator fixes; with ``num_accepted_tokens``, each sequence's tokens in turn,
        equal-length or packed, a slot a token.
        """
        heads = head_count_refusal(call.heads, call.value_heads)
        if heads is not None:
            return heads
        unsupported = [
            name
            for name, present in (
                ("packed varlen", call.varlen and not call.rollback),
                (
                    "K and V other than matching 64 or 128",
                    call.dim_k != call.dim_v or call.dim_k not in (64, 128),
                ),
                ("T other than 1", call.seq_len != 1 and not call.rollback),
            )
            if present
        ]
        return "does not support " + ", ".join(unsupported) if unsupported else None

    @classmethod
    def entry_for(cls, call: GDNCall) -> Entry:
        index = call.device.index if call.device is not None else None
        # A packed call's token count is runtime data; an equal-length one's is built in.
        tokens = 1 if call.varlen else call.seq_len
        identity = (
            call.num_sequences,
            call.heads,
            call.value_heads,
            call.dim_k,
            call.scale,
            call.dtype,
            call.has_initial_state,
            call.state_v_first,
            call.l2norm,
            call.gate_in_kernel,
            call.beta_sigmoid,
            call.allow_neg_eigval,
            call.pooled,
            call.rollback,
            call.varlen,
            tokens,
            call.drafts,
            index,
        )
        return identity, lambda: cls(
            batch=call.num_sequences,
            heads=call.heads,
            value_heads=call.value_heads,
            dim=call.dim_k,
            scale=call.scale,
            dtype=call.dtype,
            has_initial_state=call.has_initial_state,
            state_v_first=call.state_v_first,
            l2norm=call.l2norm,
            gate_in_kernel=call.gate_in_kernel,
            beta_sigmoid=call.beta_sigmoid,
            allow_neg_eigval=call.allow_neg_eigval,
            pooled=call.pooled,
            rollback=call.rollback,
            varlen=call.varlen,
            tokens=tokens,
            drafts=call.drafts,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        heads: int,
        value_heads: int,
        dim: int,
        scale: float,
        dtype: torch.dtype,
        has_initial_state: bool,
        state_v_first: bool,
        l2norm: bool,
        gate_in_kernel: bool,
        beta_sigmoid: bool,
        allow_neg_eigval: bool,
        pooled: bool = False,
        rollback: bool = False,
        varlen: bool = False,
        tokens: int = 1,
        drafts: int = 0,
        *,
        device_index: int | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.heads = heads
        self.value_heads = value_heads
        self.dim = dim
        self.scale = scale
        self.dtype = dtype
        self.has_initial_state = has_initial_state
        self.state_v_first = state_v_first
        self.gate_in_kernel = gate_in_kernel
        self.l2norm = l2norm
        self.beta_sigmoid = beta_sigmoid
        self.allow_neg_eigval = allow_neg_eigval
        self.pooled = pooled
        self.rollback = rollback
        self.varlen = varlen
        self.tokens = tokens
        self.drafts = drafts
        self.init_config()
        self.program(self._compile)
        device = (
            torch.device("cuda", device_index) if device_index is not None else torch.device("cuda")
        )
        # The parameters this build leaves unread are declared one element wide.
        self._unread = torch.empty(1, dtype=torch.float32, device=device)
        self._unread_offsets = torch.empty(1, dtype=torch.int64, device=device)

    def _compile(self, cfg: dict):
        """The decode program in the block shape *cfg* states."""
        return delta_decode_tl(
            self.batch,
            self.heads,
            self.value_heads,
            self.dim,
            self.scale,
            self.dtype_str,
            gated=True,
            gate_in_kernel=self.gate_in_kernel,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            l2norm=self.l2norm,
            has_initial_state=self.has_initial_state,
            state_v_first=self.state_v_first,
            threads=cfg["threads"],
            lane_group=cfg["lane_group"],
            pooled=self.pooled,
            rollback=self.rollback,
            varlen=self.varlen,
            tokens=self.tokens,
            drafts=self.drafts,
        )()

    @property
    def default_config(self) -> dict:
        """The block shape this call's grid and state layout call for."""
        if (
            self.build_target.arch == 90
            and self.dtype == torch.bfloat16
            and self.batch == 1
            and self.value_heads == 16
            and self.dim == 128
            and not self.state_v_first
            and not self.l2norm
            and not self.pooled
            and not self.varlen
            and not self.rollback
        ):
            return {"threads": 64, "lane_group": 4}
        return decode_launch(
            self.batch, self.value_heads, self.dim, self.state_v_first, self.build_target.sm_count
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cu_seqlens_cpu
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
        if self.has_initial_state and initial_state is None:
            raise ValueError("the build reads initial_state, but the call passed none")
        state = initial_state if self.has_initial_state else self._unread
        gate_params = (A_log, dt_bias) if self.gate_in_kernel else (self._unread, self._unread)
        program = self.program(self._compile)
        if self.rollback:
            offsets = cu_seqlens if self.varlen else self._unread_offsets
            return program(
                q, k, v, g, beta, *gate_params, state, state_indices, num_accepted_tokens, offsets
            ), state
        if self.pooled:
            # The pool is advanced in place and returned as the final state.
            return program(q, k, v, g, beta, *gate_params, state, state_indices), state
        return program(q, k, v, g, beta, *gate_params, state)


@target_cache(maxsize=32)
@tilelang.jit(out_idx=[6, 7], pass_configs={"tl.enable_fast_math": True})
def _gdn_decode_columns(batch_size, heads, scale, neg_eigval):
    width, threads = 16, 64
    vector = 2

    @T.prim_func
    def kernel(
        Q: T.Tensor((batch_size, 1, heads, 128), "bfloat16"),
        K: T.Tensor((batch_size, 1, heads, 128), "bfloat16"),
        V: T.Tensor((batch_size, 1, heads, 128), "bfloat16"),
        G: T.Tensor((batch_size, 1, heads), "bfloat16"),
        B: T.Tensor((batch_size, 1, heads), "bfloat16"),
        S: T.Tensor((batch_size, heads, 128, 128), "float32"),
        O: T.Tensor((batch_size, 1, heads, 128), "bfloat16"),
        F: T.Tensor((batch_size, heads, 128, 128), "float32"),
    ):
        with T.Kernel(128 // width, batch_size * heads, threads=threads) as (tile, head):
            q = T.alloc_fragment((128,), "float32")
            k = T.alloc_fragment((128,), "float32")
            qs = T.alloc_fragment((128,), "float32")
            ks = T.alloc_fragment((128,), "float32")
            qn = T.alloc_fragment((1,), "float32")
            kn = T.alloc_fragment((1,), "float32")
            state = T.alloc_fragment((128, width), "float32")
            products = T.alloc_fragment((128, width), "float32")
            projection = T.alloc_fragment((width,), "float32")
            delta = T.alloc_fragment((width,), "float32")
            matrix_layout = T.Fragment(
                (128, width),
                forward_thread_fn=lambda i, j: (i * (width // vector) + j // vector) % threads,
            )
            T.annotate_layout({state: matrix_layout, products: matrix_layout})
            batch = head // heads
            h = head % heads
            decay = T.exp(T.cast(G[batch, 0, h], "float32"))
            beta = T.sigmoid(T.cast(B[batch, 0, h], "float32")) * (2.0 if neg_eigval else 1.0)
            for i in T.Parallel(128):
                q[i] = Q[batch, 0, h, i]
                k[i] = K[batch, 0, h, i]
                qs[i] = q[i] * q[i]
                ks[i] = k[i] * k[i]
            T.reduce_sum(qs, qn, dim=0)
            T.reduce_sum(ks, kn, dim=0)
            for i in T.Parallel(128):
                q[i] *= T.rsqrt(qn[0] + 1e-6)
                k[i] *= T.rsqrt(kn[0] + 1e-6)
            for i, j in T.Parallel(128, width):
                state[i, j] = S[batch, h, i, tile * width + j] * decay
                products[i, j] = state[i, j] * k[i]
            T.reduce_sum(products, projection, dim=0, batch=vector)
            for j in T.Parallel(width):
                delta[j] = beta * (
                    T.cast(V[batch, 0, h, tile * width + j], "float32") - projection[j]
                )
            for i, j in T.Parallel(128, width):
                state[i, j] = state[i, j] + k[i] * delta[j]
                F[batch, h, i, tile * width + j] = state[i, j]
                products[i, j] = state[i, j] * q[i]
            T.reduce_sum(products, projection, dim=0, batch=vector)
            for j in T.Parallel(width):
                O[batch, 0, h, tile * width + j] = projection[j] * scale

    return kernel


class GDNColumnDecodeKernel(Kernel, GDNFwdInterface):
    """Keep a narrow column tile of the FP32 state in registers on Hopper."""

    supported_archs = [90]
    preferred_over = frozenset({"gdn_dense_decode"})

    @classmethod
    def refusal(cls, call: GDNCall) -> str | None:
        if (
            call.pooled
            or call.varlen
            or call.rollback
            or call.state_v_first
            or call.seq_len != 1
            or not call.has_initial_state
            or call.gate_in_kernel
            or not call.l2norm
            or not call.beta_sigmoid
        ):
            return (
                "requires one dense token, batch-ordered K-major state, Q/K norm and beta sigmoid"
            )
        if (
            call.dtype != torch.bfloat16
            or (call.dim_k, call.dim_v) != (128, 128)
            or call.heads != call.value_heads
            or not 1 <= call.batch * call.heads <= 64
        ):
            return "requires at most 64 BF16 query/value heads across the batch with width 128"
        return head_count_refusal(call.heads)

    @classmethod
    def entry_for(cls, call: GDNCall) -> Entry:
        return call, lambda: cls(call)

    def __init__(self, call: GDNCall):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call

    def _compile(self, cfg):
        call = self.call
        return _gdn_decode_columns(call.batch, call.heads, call.scale, call.allow_neg_eigval)

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        A_log=None,
        dt_bias=None,
        state_indices=None,
        num_accepted_tokens=None,
    ):
        return self.program(self._compile)(q, k, v, g, beta, initial_state)
