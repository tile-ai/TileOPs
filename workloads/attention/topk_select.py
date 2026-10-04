import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


class TopKSelectWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        kv_group: int,
        topk: int,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        index_score = torch.randn(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            dtype=self.in_dtype,
            device=run_device(),
        )
        starts = torch.zeros(self.batch, self.seq_len, dtype=self.out_dtype, device=run_device())
        ends = (
            torch.ones(self.batch, self.seq_len, dtype=self.out_dtype, device=run_device())
            * self.seq_len_kv
        )
        return index_score, starts, ends

    def ref_program(
        self, index_score: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(index_score.shape[2], device=index_score.device)[
            None, None, :, None
        ]
        valid = (positions >= starts[:, :, None, None]) & (positions < ends[:, :, None, None])
        masked = index_score.masked_fill(~valid, -float("inf"))
        indexes = torch.topk(masked, self.topk, dim=2).indices
        in_window = valid.expand_as(index_score).gather(2, indexes)
        indexes = torch.where(in_window, indexes, index_score.shape[2])
        return indexes.permute(0, 1, 3, 2).to(self.out_dtype)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        def validate(got, expected):
            assert got.shape == expected.shape and got.dtype == expected.dtype
            scores, starts, ends = inputs
            padding = got == scores.shape[2]
            assert (
                padding | ((got >= starts[:, :, None, None]) & (got < ends[:, :, None, None]))
            ).all()
            assert torch.equal(padding.sum(-1), (expected == scores.shape[2]).sum(-1))
            ordered = got.sort(-1).values
            assert (
                (ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] == scores.shape[2])
            ).all(), "duplicate selected index"
            scores = torch.nn.functional.pad(scores.movedim(2, -1), (0, 1), value=-float("inf"))
            torch.testing.assert_close(
                scores.gather(-1, got.long()).sort(-1).values,
                scores.gather(-1, expected.long()).sort(-1).values,
                rtol=0,
                atol=0,
            )

        return Custom(validate, "selected values, unique indices and exact padding")


class TopKSelectCall(CallWorkload, TopKSelectWorkload):
    """A manifest call of TopKSelectFwdOp; the row's generators give the windows."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix = call.ix
        TopKSelectWorkload.__init__(
            self, ix["B"], ix["S"], ix["S_kv"], ix["G"], ix["topk"], torch.float32, torch.int32
        )

    gen_inputs = CallWorkload.gen_inputs
