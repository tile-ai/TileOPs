from typing import Tuple

import torch

from benchmarks.benchmark import Benchmark
from top.ops import TopkSelectorOp, Op


class TopkSelectorBenchmark(Benchmark):
    op_type = TopkSelectorOp

    def __init__(
        self,
        batch: int,
        seq_len: int,
        topk: int,
        in_dtype: str,
        out_dtype: str,
        # index_score: torch.float32,
        # index: torch.int32,
        # starts: torch.int32,
        # ends: torch.int32
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        # self.index_score = index_score
        # self.index = index
        # self.starts = starts
        # self.ends = ends

    @property
    def total_flops(self) -> float:
        return None

    @property
    def total_memory(self) -> float:
        # index_score: batch, seq_len
        # index: batch, topk
        # starts: batch
        # ends: batch
        index_score_memory = self.batch * self.seq_len * self.in_dtype.itemsize
        index_memory = self.batch * self.topk * self.out_dtype.itemsize
        starts_memory = self.batch * self.out_dtype.itemsize
        ends_memory = self.batch * self.out_dtype.itemsize
        return index_score_memory + index_memory + starts_memory + ends_memory

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(self.batch, self.seq_len, dtype=self.in_dtype, device="cuda")
        starts = torch.zeros(self.batch, dtype=self.out_dtype, device="cuda")
        ends = torch.ones(self.batch, dtype=self.out_dtype, device="cuda") * self.seq_len
        return index_score, starts, ends

    def ref_program(self, index_score, starts, ends):
        indexes_ref = torch.topk(index_score, self.topk, dim=-1)[1]
        return indexes_ref

    def check(self,
              op: Op,
              *inputs: Tuple[torch.Tensor],
              atol: float = 1e-2,
              rtol: float = 1e-2) -> None:
        """Check the correctness of the op"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"

        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            ref_np = outputs_ref[i].cpu().to(torch.int32).numpy()
            trt_np = outputs[i].cpu().to(torch.int32).numpy()

            ref_list = ref_np.flatten().tolist()
            trt_list = trt_np.flatten().tolist()

            set_ref = set(ref_list)
            set_trt = set(trt_list)
            intersection = set_ref & set_trt
            assert len(intersection) / len(
                set_ref) == 1.0, "outputs[{i}] is not close to outputs_ref[{i}]"

        print(f"All checks passed for {op.__class__.__name__}.✅")

    def check_fn(self,
                 fn: callable,
                 *inputs: Tuple[torch.Tensor],
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 grad: bool = True) -> None:
        """Check the correctness of the function and layer"""
        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        if not grad:
            with torch.no_grad():
                outputs = fn(*inputs)
        else:
            output = fn(*inputs)
            loss = output.sum()
            loss.backward()
            outputs = []
            outputs.append(output)
            for inp in inputs:
                outputs.append(inp.grad)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            ref_np = outputs_ref[i].cpu().to(torch.int32).numpy()
            trt_np = outputs[i].cpu().to(torch.int32).numpy()

            ref_list = ref_np.flatten().tolist()
            trt_list = trt_np.flatten().tolist()

            set_ref = set(ref_list)
            set_trt = set(trt_list)
            intersection = set_ref & set_trt
            assert len(intersection) / len(
                set_ref) == 1.0, "outputs[{i}] is not close to outputs_ref[{i}]"

        print(f"All checks passed for {fn.__class__.__name__}.✅")
