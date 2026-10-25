"""How a class name is spelled, and the torch operator name derived from it.

`ABBREVIATIONS` is the one table of abbreviations a class name may hold. The naming lint
rejects every other spelling of each, and `operator_name` keeps each one a single word.
Imports nothing outside the standard library, so the lint reads it without torch.
"""

import re

__all__ = ["ABBREVIATIONS", "operator_name"]

# Each abbreviation's one spelling, mapped to a pattern matching every other spelling of it.
ABBREVIATIONS = {
    "BMM": r"Bmm(?![a-z])",
    "C2C": r"C2c",
    "DSA": r"Dsa",
    "ELU": r"(?<![A-Z])Elu(?![a-z])",
    "FFT": r"Fft",
    "FP8": r"Fp8",
    "GDN": r"Gdn",
    "GELU": r"Gelu(?![a-z])",
    "GEMM": r"Gemm(?![a-z])(?!WarpPolicy)",
    "GLA": r"Gla",
    "GQA": r"Gqa",
    "INT4": r"Int4",
    "INT8": r"Int8",
    "KDA": r"Kda",
    "KV": r"Kv(?=[A-Z0-9]|$)",
    "LogSumExp": r"Logsumexp",
    "MHA": r"Mha",
    "MHC": r"Mhc",
    "MinP": r"Minp",
    "MLA": r"Mla",
    "MLP": r"Mlp",
    "MMA": r"Mma",
    "MoE": r"Moe",
    "NSA": r"Nsa",
    "PReLU": r"P[Rr]elu(?![a-z])",
    "ReLU": r"(?<![Pp])Relu(?![a-z])",
    "RMS": r"Rms(?=Norm)",
    "RoPE": r"Rope",
    "SELU": r"Selu(?![a-z])",
    "SiLU": r"Silu(?![a-z])",
    "SSD": r"Ssd",
    "TMA": r"Tma",
    "TopK": r"Topk",
    "TopP": r"Topp",
    "W4A16": r"W4a16",
    "W4A8": r"W4a8",
    "W8A8": r"W8a8",
    "WGMMA": r"Wgmma",
    "WS": r"Ws(?=[A-Z0-9]|$)",
    "YaRN": r"Yarn",
}

# Longest first, so an abbreviation inside another (`ReLU` in `PReLU`) never splits it.
_BY_LENGTH = sorted(ABBREVIATIONS, key=len, reverse=True)


def operator_name(family: str, class_name: str) -> str:
    """``("norm", "RMSNormFwdOp")`` -> ``"norm_rms_norm_fwd"``; a class whose own name already
    opens with the family, such as ``MoEPrePermuteFwdOp``, names it once.

    An abbreviation in `ABBREVIATIONS` stays one word.
    """
    held: dict[str, str] = {}
    name = class_name
    for i, abbreviation in enumerate(_BY_LENGTH):
        if abbreviation in name:
            # A placeholder the case-split rules leave alone: one capital, then lowercase.
            holder = f"Zz{i:02d}zz"
            held[holder.lower()] = abbreviation.lower()
            name = name.replace(abbreviation, holder)
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", spaced).lower().removesuffix("_op")
    for holder, abbreviation in held.items():
        stem = stem.replace(holder, abbreviation)
    if stem == family or stem.startswith(f"{family}_"):
        return stem
    return f"{family}_{stem}"
