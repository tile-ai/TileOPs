# Issue Title Format

**Format:** `[TYPE][COMPONENT] short description in lowercase`

- **TYPE**: a key of `ISSUE_TO_COMMIT_TYPE` in [`.claude/conventions/types.sh`](../../.claude/conventions/types.sh), which is canonical. Read it rather than a copy — a copy here would drift.
- **COMPONENT**: mandatory — kernel name or subsystem (GEMV, GEMM, FLASH_ATTN, CI, TOOLING, …)
- Max 80 characters total, description in lowercase
- Do NOT use Conventional Commits style (`feat(scope): …`)

Examples: `[FEAT][GEMV] add batched forward pass` · `[BUG][FLASH_ATTN] fix bf16 overflow in softmax` · `[PERF][GEMM] optimize shared memory layout for H100`
