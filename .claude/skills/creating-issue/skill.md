---
name: creating-issue
description: Create a high-quality issue which conforms to the rules of TileOPs development
---

# Creating Issues — Guidelines and Lessons Learned

> Best practices for filing GitHub issues in TileOps, based on experience from working with this repository.

______________________________________________________________________

## 1. Title Format

TileOps uses a `[TYPE][COMPONENT]` tag prefix — **not** Conventional Commits style.

```
[TYPE][COMPONENT] short description in lowercase

TYPE values : PERF, BUG, FEAT, REFACTOR, TEST, DOCS, META
COMPONENT   : kernel name or subsystem — e.g., GEMV, GEMM, FLASH_ATTN, FLASH_DECODE, CI
```

**Examples**:

```
[PERF][GEMV] fix uncoalesced memory access on H200
[BUG][FLASH_DECODE] incorrect output with paged KV cache
[FEAT][GROUPED_GEMM] add FP8 support
[META][TOOLING] standardize issue title format in contribution guidelines
```

**Do NOT use**:

```
perf(gemv): fix uncoalesced memory access    ← Conventional Commits style, wrong
fix: gemv stride access                      ← too vague, no TYPE/COMPONENT tags
```

______________________________________________________________________

## 2. Issue Body Structure

```markdown
## Problem
Clear description of the observed issue. Include:
- Which file/function is affected
- Why it is wrong or suboptimal (root cause if known)
- Code snippet showing the problematic pattern

## Proposed Fix  (for PERF/BUG issues)
- Concrete change description with before/after code snippets
- Expected improvement (quantified where possible)

## Expected Impact  (for PERF issues)
- Table of optimizations and estimated gains
- Affected benchmark shapes

## Next Steps
- Whether a PR will follow, and roughly what it will contain
```

______________________________________________________________________

## 3. Language

All issue titles and bodies must be written in **English**.

______________________________________________________________________

## 4. When to File an Issue Before a PR

Always file an issue first when:

- The change involves a non-trivial performance optimization
- The root cause analysis is worth documenting separately from the code diff
- You want upstream visibility before investing in a full PR

For trivial typo fixes or single-line changes, a PR without a prior issue is fine.

______________________________________________________________________

## 5. AI Tooling Note

When asking an AI assistant (e.g., Claude Code) to file an issue, always specify:

1. The exact title format: `[TYPE][COMPONENT] description`
1. The required language: English
1. Whether a PR will follow (affects the "Next Steps" section)

Without explicit instructions, AI tools may default to Conventional Commits style titles or mix languages.

______________________________________________________________________

## 6. Case Log

| Date       | Issue                                                                 | Note                                                                                                     |
| ---------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 2026-02-27 | [[PERF][GEMV] #232](https://github.com/tile-ai/TileOPs/issues/232)    | GEMV uncoalesced B access on H200; title initially filed as `perf(gemv): ...`, corrected post-submission |
| 2026-02-27 | [[META][TOOLING] #233](https://github.com/tile-ai/TileOPs/issues/233) | Upstream issue requesting formal title format documentation in CONTRIBUTING.md                           |
