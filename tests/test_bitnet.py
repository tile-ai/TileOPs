import argparse
import torch
from top import Bitnet_158_int8xint2_kernel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=int, default=1, help='')
    parser.add_argument('--N', type=int, default=256, help='')
    parser.add_argument('--K', type=int, default=256, help='')
    parser.add_argument('--in_dtype', type=str, default="int8", help='')
    parser.add_argument('--out_dtype', type=str, default="int32", help='')
    parser.add_argument('--accum_dtype', type=str, default="int32", help='')
    parser.add_argument('--fast_decoding', type=bool, default=True, help='')

    args = parser.parse_args()
    M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding = args.M, args.N, args.K, args.in_dtype, args.out_dtype, args.accum_dtype, args.fast_decoding

    A = torch.randint(0, 4, (M, K), device="cuda", dtype=getattr(torch, in_dtype))
    B = torch.randint(0, 2, (N, K), device="cuda", dtype=getattr(torch, in_dtype))
    C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

    bitnet_158_int8xint2 = Bitnet_158_int8xint2_kernel(
        M, N, K, in_dtype, out_dtype, accum_dtype, fast_decoding)
    C1 = bitnet_158_int8xint2.prefill(A, B, C)
    C2 = bitnet_158_int8xint2.decode(A, B, C)
    print(C1)
    print(C2)
    bitnet_158_int8xint2.check(A, B, C, "prefill")
    bitnet_158_int8xint2.check(A, B, C, "decode")


if __name__ == "__main__":
    main()
