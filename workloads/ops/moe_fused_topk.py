import torch

from workloads.base import WorkloadBase


class FusedTopKTest(WorkloadBase):
    def __init__(self, num_tokens, num_experts, top_k, scoring_func, renormalize, dtype):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        return torch.randn(self.num_tokens, self.num_experts, dtype=self.dtype, device="cuda")

def _ref_fused_topk(
    gating_output: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = gating_output.to(torch.float32)
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"Unknown scoring_func: {scoring_func}")

    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids.int()
