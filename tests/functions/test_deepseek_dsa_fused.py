import pytest
import torch

from benchmarks import DeepSeekDSAFusedBenchmark
from top.functions import DeepSeekDSAFusedFunc
from top.utils import str2dtype
from typing import Any


@pytest.mark.parametrize(
    ("batch", "seq_len", "seq_len_kv", "heads", "dim", "index_dim", "topk", "dim_tail", "stride_kv",
     "group_kv", "q_start_index_s", "quant_in_dtype", "clean_logits", "in_dtype", "out_dtype",
     "sm_scale", "is_causal", "dsa_dtype", "tune"),
    [
        (1, 512, 1024, 64, 256, 32, 128, 32, 1, 1, 512, torch.float16, True, 'float16', 'int32',
         None, True, 'float16', False),

        # (1,1024,2048,128,512,64,128,64,1,1,1024,'float16',True,'float32','int32',None,True,'float16',False),
        #  (1, 128, 64, 8, 64, 256, 10, 16, 32, 8, 0, torch.float16, True, "float16", "int32", None, True,
        #   torch.float16, False),
        #  (2, 256, 128, 16, 128, 512, 20, 64, 64, 16, 1, torch.float32, False, "float32", "int32", 1.0,
        #   False, torch.float32, True),
        #  (4, 512, 256, 32, 256, 1024, 50, 128, 128, 32, 0, torch.float16, True, "float16", "int64",
        #   None, True, torch.float16, False),
        #  (8, 1024, 512, 64, 512, 2048, 100, 256, 256, 64, 0, torch.bfloat16, False, "bfloat16",
        #   "int32", 0.5, False, torch.float16, True),
        #  (16, 128, 128, 4, 32, 128, 10, 16, 16, 4, 0, torch.float32, True, "float32", "int64", None,
        #   True, torch.float32, False),
        #  (32, 512, 256, 8, 128, 512, 50, 128, 64, 8, 1, torch.float16, False, "float16", "int32", None,
        #   True, torch.bfloat16, True),
        #  (64, 256, 128, 16, 256, 1024, 20, 128, 128, 16, 1, torch.bfloat16, True, "bfloat16", "int32",
        #   0.8, False, torch.float32, False),
        #  (1, 1024, 512, 32, 512, 2048, 10, 256, 128, 32, 1, torch.float32, False, "float32", "int64",
        #   None, True, torch.float32, True),
        #  (2, 128, 64, 8, 64, 256, 5, 16, 32, 4, 0, torch.float16, True, "float16", "int32", 0.5, True,
        #   torch.bfloat16, False),
        #  (4, 256, 128, 16, 128, 512, 25, 128, 64, 8, 1, torch.float32, False, "float32", "int64", 1.0,
        #   True, torch.float32, True)
    ])
def test_deepseek_dsa_fused(
        batch: int,
        seq_len: int,
        seq_len_kv: int,
        heads: int,
        dim: int,
        index_dim: int,
        topk: int,
        dim_tail: int,
        stride_kv: int,
        group_kv: int,
        q_start_index_s: int,
        # quant_in_dtype: torch.dtype = torch.float16,
        # clean_logits: bool = True,
        # in_dtype: str = "float16",
        # out_dtype: str = "int32",
        # sm_scale: Any = None,
        # is_causal: bool = True,
        # dsa_dtype: torch.dtype = torch.float16,
        # tune: bool = False,
        quant_in_dtype: torch.dtype,
        clean_logits: bool,
        in_dtype: str,
        out_dtype: str,
        sm_scale: Any,
        is_causal: bool,
        dsa_dtype: torch.dtype,
        tune: bool):
    """Test DeepSeekDSAFusedFunc with various configurations"""
    torch.cuda.empty_cache()

    # Initialize function
    fn = DeepSeekDSAFusedFunc(
        seq_len_kv=seq_len_kv,
        index_dim=index_dim,
        seq_len=seq_len,
        heads=heads,
        batch=batch,
        topk=topk,
        dim=dim,
        dim_tail=dim_tail,  # Default tail dimension
        stride_kv=1,
        group_kv=1,
        q_start_index_s=q_start_index_s,
        quant_in_dtype=quant_in_dtype,
        clean_logits=True,
        in_dtype=str2dtype.get(in_dtype),
        out_dtype=str2dtype.get(out_dtype),
        sm_scale=sm_scale,
        is_causal=is_causal,
        dsa_dtype=dsa_dtype,
        tune=tune)

    # Initialize benchmark
    benchmark = DeepSeekDSAFusedBenchmark(
        seq_len_kv=seq_len_kv,
        index_dim=index_dim,
        seq_len=seq_len,
        heads=heads,
        batch=batch,
        topk=topk,
        dim=dim,
        dim_tail=dim_tail,  # Default tail dimension
        stride_kv=1,
        group_kv=1,
        q_start_index_s=q_start_index_s,
        quant_in_dtype=quant_in_dtype,
        clean_logits=True,
        in_dtype=str2dtype.get(in_dtype),
        out_dtype=str2dtype.get(out_dtype),
        sm_scale=sm_scale,
        is_causal=is_causal,
        dsa_dtype=dsa_dtype,
        tune=tune)

    # Generate inputs
    inputs = benchmark.gen_inputs()

    try:
        print("Testing FusedDeepSeekSparseAttentionFunc...")
        benchmark.check_fn(fn, inputs, grad=False)
        print("✅ FusedDeepSeekSparseAttentionFunc test passed")
    except Exception as e:
        print(f"❌ FusedDeepSeekSparseAttentionFunc test failed: {e}")
        raise


if __name__ == "__main__":
    # test_deepseek_dsa_fused(1, 128, 64, 8, 64, 256, 10, 16, 32, 8, 0, torch.float16, True,
    #                         "float16", "int32", None, True, torch.float16, False),
    # test_deepseek_dsa_fused(2, 256, 128, 16, 128, 512, 20, 64, 64, 16, 1, torch.float32, False,
    #                         "float32", "int32", 1.0, False, torch.float32, True),
    # test_deepseek_dsa_fused(4, 512, 256, 32, 256, 1024, 50, 128, 128, 32, 0, torch.float16, True,
    #                         "float16", "int64", None, True, torch.float16, False),
    # test_deepseek_dsa_fused(8, 1024, 512, 64, 512, 2048, 100, 256, 256, 64, 0, torch.bfloat16,
    #                         False, None, "bfloat16", "int32", 0.5, False, torch.float16, True),
    # test_deepseek_dsa_fused(16, 128, 128, 4, 32, 128, 10, 16, 16, 4, 0, torch.float32, True,
    #                         "float32", "int64", None, True, torch.float32, False),
    # test_deepseek_dsa_fused(32, 512, 256, 8, 128, 512, 50, 128, 64, 8, 1, torch.float16, False,
    #                         "float16", "int32", None, True, torch.bfloat16, True),
    # test_deepseek_dsa_fused(64, 256, 128, 16, 256, 1024, 20, 128, 128, 16, 1, torch.bfloat16, True,
    #                         "bfloat16", "int32", 0.8, False, torch.float32, False),
    # test_deepseek_dsa_fused(1, 1024, 512, 32, 512, 2048, 10, 256, 128, 32, 1, torch.float32, False,
    #                         "float32", "int64", None, True, torch.float32, True),
    # test_deepseek_dsa_fused(2, 128, 64, 8, 64, 256, 5, 16, 32, 4, 0, torch.float16, True, "float16",
    #                         "int32", 0.5, True, torch.bfloat16, False),
    # test_deepseek_dsa_fused(4, 256, 128, 16, 128, 512, 25, 128, 64, 8, 1, torch.float32, False,
    #                         "float32", "int64", 1.0, True, torch.float32, True)
    test_deepseek_dsa_fused(1, 512, 1024, 64, 256, 32, 128, 32, 1, 1, 512, torch.float16, True,
                            'float16', 'int32', None, True, 'float16', False)
