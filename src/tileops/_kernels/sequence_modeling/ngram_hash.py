"""EOS-bounded n-gram hashing and a subsequent, race-free context update."""

from abc import abstractmethod
from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import Entry, Kernel, KernelInterface

__all__ = ["NGramHashCall", "NGramHashFwdInterface", "NGramHashKernel"]


@dataclass(frozen=True)
class NGramHashCall(CallSpec):
    """Static hash and context layout facts; metadata values stay on device."""

    width: int = 0
    row_stride: int = 0
    col_stride: int = 0
    order: int = 0
    heads: int = 0
    eos: int = 0
    packed: bool = False
    slotted: bool = False
    rollback: bool = False


class NGramHashFwdInterface(KernelInterface):
    """Hash tokens and update the addressed context rows in place."""

    request = NGramHashCall

    @abstractmethod
    def forward(
        self,
        input_ids,
        context,
        multipliers,
        primes,
        offsets,
        cu_seqlens=None,
        context_indices=None,
        num_accepted_tokens=None,
    ):
        """Return int64 ids with a final heads axis on call.device.

        Read-only tensors are contiguous. Context retains the strides in the call
        and is written in place. Optional int32 metadata has the manifest shapes;
        context indices are distinct. Hashing reads the pre-call history, then
        the context commits the ordinary tail or speculative history window.
        """


@target_cache(maxsize=32)
def _ngram_hash_kernel(width, row_stride, col_stride, order, packed, slotted, rollback, heads, eos):
    @tilelang.jit(out_idx=[])
    def build(block, threads):
        batch = T.dynamic("batch")
        length = 0 if packed else T.dynamic("length")
        total = T.dynamic("total") if packed else batch * length
        slots = T.dynamic("slots")

        @T.prim_func
        def hash_ids(
            Tokens: T.Tensor((total,) if packed else (batch, length), "int64"),
            Context: T.StridedTensor((slots, width), (row_stride, col_stride), "int64"),
            Multipliers: T.Tensor((order,), "int64"),
            Primes: T.Tensor((heads,), "int64"),
            Offsets: T.Tensor((heads,), "int64"),
            Cu: T.Tensor((batch + 1,) if packed else (total,), "int32" if packed else "int64"),
            Slots: T.Tensor((batch,) if slotted else (total,), "int32" if slotted else "int64"),
            Accepted: T.Tensor(
                (batch,) if rollback else (total,), "int32" if rollback else "int64"
            ),
            Output: T.Tensor((total, heads), "int64"),
        ):
            with T.Kernel(T.ceildiv(total * heads, block), threads=threads) as bx:
                for x in T.Parallel(block):
                    linear = bx * block + x
                    if linear < total * heads:
                        token = linear // heads
                        head = linear % heads
                        seq = T.alloc_var("int32", init=0)
                        begin = T.alloc_var("int32", init=0)
                        if packed:
                            hi = T.alloc_var("int32", init=batch)
                            while seq < hi:
                                mid = (seq + hi) // 2
                                if Cu[mid + 1] <= token:
                                    seq = mid + 1
                                else:
                                    hi = mid
                            begin = Cu[seq]
                        else:
                            seq = token // length
                            begin = seq * length
                        slot = Slots[seq] if slotted else seq
                        start = Accepted[seq] - 1 if rollback else 0
                        mixed = T.alloc_var("uint64", init=0)
                        blocked = T.alloc_var("bool", init=False)
                        for i in T.unroll(order):
                            value = T.alloc_var("int64", init=eos)
                            if i == 0 or not blocked:
                                if token - begin >= i:
                                    if packed:
                                        value = Tokens[token - i]
                                    else:
                                        value = Tokens[seq, token - begin - i]
                                else:
                                    value = Context[slot, start + order - 1 + token - begin - i]
                            # Only a previous EOS blocks history; the current EOS still hashes.
                            if i > 0 and value == eos:
                                blocked = True
                            if i < 2 + head // (heads // (order - 1)):
                                mixed = mixed ^ (
                                    T.cast(value, "uint64") * T.cast(Multipliers[i], "uint64")
                                )
                        signed_hash = T.cast(mixed, "int64")
                        remainder = T.truncmod(signed_hash, Primes[head])
                        positive = T.if_then_else(
                            remainder < 0, remainder + Primes[head], remainder
                        )
                        Output[token, head] = T.cast(
                            T.cast(positive, "uint64") + T.cast(Offsets[head], "uint64"), "int64"
                        )

        return hash_ids

    return build


@target_cache(maxsize=32)
def _ngram_context_kernel(width, row_stride, col_stride, order, packed, slotted, rollback):
    @tilelang.jit(out_idx=[])
    def build(threads):
        batch = T.dynamic("batch")
        length = 0 if packed else T.dynamic("length")
        total = T.dynamic("total") if packed else batch * length
        slots = T.dynamic("slots")

        @T.prim_func
        def update(
            Tokens: T.Tensor((total,) if packed else (batch, length), "int64"),
            Context: T.StridedTensor((slots, width), (row_stride, col_stride), "int64"),
            Cu: T.Tensor((batch + 1,) if packed else (total,), "int32" if packed else "int64"),
            Slots: T.Tensor((batch,) if slotted else (total,), "int32" if slotted else "int64"),
            Accepted: T.Tensor(
                (batch,) if rollback else (total,), "int32" if rollback else "int64"
            ),
        ):
            with T.Kernel(batch, threads=threads) as b:
                begin = Cu[b] if packed else b * length
                end = Cu[b + 1] if packed else (b + 1) * length
                slot = Slots[b] if slotted else b
                start = Accepted[b] - 1 if rollback else 0
                old = T.alloc_shared((order - 1,), "int64")
                if end > begin:
                    for i in T.Parallel(order - 1):
                        old[i] = Context[slot, start + i]
                    # A shifted read and write may overlap in the same row.
                    T.sync_threads()
                    for j in T.Parallel(width if rollback else order - 1):
                        source = j + 1 if rollback else end - begin + j
                        if source < order - 1 + end - begin:
                            if source < order - 1:
                                Context[slot, j] = old[source]
                            else:
                                if packed:
                                    Context[slot, j] = Tokens[begin + source - (order - 1)]
                                else:
                                    Context[slot, j] = Tokens[b, source - (order - 1)]

        return update

    return build


class NGramHashKernel(Kernel, NGramHashFwdInterface):
    """Hash all tokens before committing any context row."""

    @classmethod
    def entry_for(cls, call: NGramHashCall) -> Entry:
        # Token counts and context pool size are runtime tensor dimensions.
        args = (
            call.width,
            call.row_stride,
            call.col_stride,
            call.order,
            call.packed,
            call.slotted,
            call.rollback,
            call.heads,
            call.eos,
        )
        device_index = call.device_index
        return (*args, device_index), lambda: cls(*args, device_index=device_index)

    def __init__(
        self,
        width,
        row_stride,
        col_stride,
        order,
        packed,
        slotted,
        rollback,
        heads,
        eos,
        *,
        device_index,
    ):
        super().__init__(device_index=device_index)
        self.heads = heads
        shared = (width, row_stride, col_stride, order, packed, slotted, rollback)
        self.kernel = _ngram_hash_kernel(*shared, heads, eos)
        self.context_kernel = _ngram_context_kernel(*shared)
        self.init_config(None)

    @property
    def default_config(self):
        # Four warps cover adjacent hash heads and amortize the metadata lookup.
        return {"block": 128, "threads": 128}

    def _compile_hash(self, cfg):
        return self.kernel(cfg["block"], cfg["threads"])

    def _compile_context(self, cfg):
        return self.context_kernel(cfg["threads"])

    def forward(
        self,
        input_ids,
        context,
        multipliers,
        primes,
        offsets,
        cu_seqlens=None,
        context_indices=None,
        num_accepted_tokens=None,
    ):
        output = torch.empty(
            (*input_ids.shape, self.heads), device=input_ids.device, dtype=torch.int64
        )
        if input_ids.numel() == 0:
            return output
        tokens = input_ids.view(-1)
        cu = tokens if cu_seqlens is None else cu_seqlens
        slots = tokens if context_indices is None else context_indices
        accepted = tokens if num_accepted_tokens is None else num_accepted_tokens
        if self.heads:
            self.program(self._compile_hash)(
                input_ids,
                context,
                multipliers,
                primes,
                offsets,
                cu,
                slots,
                accepted,
                output.view(-1, self.heads),
            )
        self.program(self._compile_context)(input_ids, context, cu, slots, accepted)
        return output
