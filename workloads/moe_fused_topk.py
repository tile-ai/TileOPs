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
