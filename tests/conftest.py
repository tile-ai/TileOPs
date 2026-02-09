import pytest
import torch


@pytest.fixture(autouse=True)
def setup() -> None:
    """全局设置函数，自动为所有测试设置随机种子"""
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
