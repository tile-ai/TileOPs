import torch

from workloads.device import run_device
from workloads.numerics import Custom
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
        fill_value: int | None = None,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.fill_value = fill_value

    def fill(self, seq_len_kv: int) -> int:
        """The index of an unselected slot for scores over *seq_len_kv* keys."""
        return seq_len_kv if self.fill_value is None else self.fill_value

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = torch.randn(
            self.batch,
            self.seq_len,
            self.seq_len_kv,
            self.kv_group,
            dtype=self.in_dtype,
            device=run_device(),
        )
        starts = torch.zeros(self.batch, self.seq_len, dtype=self.out_dtype, device=run_device())
        ends = torch.full_like(starts, self.seq_len_kv)
        return logits, starts, ends

    def ref_program(
        self, logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(logits.shape[2], device=logits.device)[None, None, :, None]
        valid = (positions >= starts[:, :, None, None]) & (positions < ends[:, :, None, None])
        masked = logits.masked_fill(~valid, -float("inf"))
        selected = min(self.topk, logits.shape[2])
        fill = self.fill(logits.shape[2])
        indexes = torch.topk(masked, selected, dim=2).indices
        in_window = valid.expand_as(logits).gather(2, indexes)
        indexes = torch.where(in_window, indexes, fill)
        indexes = torch.nn.functional.pad(indexes, (0, 0, 0, self.topk - selected), value=fill)
        return indexes.permute(0, 1, 3, 2).to(self.out_dtype)

    def verification(self, *inputs):
        def validate(got, expected):
            assert got.shape == expected.shape and got.dtype == expected.dtype
            scores, starts, ends = inputs
            seq_len_kv = scores.shape[2]
            fill = self.fill(seq_len_kv)
            padding = got == fill
            assert (
                padding | ((got >= starts[:, :, None, None]) & (got < ends[:, :, None, None]))
            ).all()
            assert torch.equal(padding.sum(-1), (expected == fill).sum(-1))
            ordered = got.sort(-1).values
            assert ((ordered[..., 1:] != ordered[..., :-1]) | (ordered[..., 1:] == fill)).all(), (
                "duplicate selected index"
            )
            # A filled slot reads the -inf past the last score.
            scores = torch.nn.functional.pad(scores.movedim(2, -1), (0, 1), value=-float("inf"))

            def picked(indices):
                indices = indices.masked_fill(indices == fill, seq_len_kv).long()
                return scores.gather(-1, indices).sort(-1).values

            torch.testing.assert_close(picked(got), picked(expected), rtol=0, atol=0)

        return Custom(validate, "selected values, unique indices and exact padding")


class TopKSelectCall(CallWorkload, TopKSelectWorkload):
    """A manifest call of TopKSelectFwdOp; the row's generators give the windows."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices = call.indices
        TopKSelectWorkload.__init__(
            self,
            indices["B"],
            indices["S"],
            indices["S_kv"],
            indices["G"],
            indices["top_k"],
            torch.float32,
            torch.int32,
            call.params["fill_value"],
        )

    gen_inputs = CallWorkload.gen_inputs
